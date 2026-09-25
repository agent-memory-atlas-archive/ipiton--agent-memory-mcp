package embedder

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"go.uber.org/zap"
)

// llamaServer answers every embeddings call with the given status and body.
func llamaServer(t *testing.T, status int, body string) *Embedder {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	e, err := New(Config{LlamaCPPBaseURL: srv.URL + "/v1", LlamaCPPModel: "granite", Dimension: 4}, zap.NewNop())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return e
}

func TestIsInputRejection(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"llama-server oversize answers 500", &providerStatusError{status: 500, body: `{"error":{"code":500,"message":"input (40003 tokens) is too large to process. increase the physical batch size (current batch size: 8192)"}}`}, true},
		{"413", &providerStatusError{status: 413, body: ""}, true},
		{"openai context length", &providerStatusError{status: 400, body: "This model's maximum context length is 8192 tokens"}, true},
		{"wrapped by the adapter", fmt.Errorf("llama.cpp request failed: %w", &providerStatusError{status: 413}), true},
		{"bad key is configuration, not input", &providerStatusError{status: 401, body: "invalid api key"}, false},
		{"server down", &providerStatusError{status: 503, body: "loading model"}, false},
		{"connection refused", errors.New("dial tcp 127.0.0.1:8090: connect: connection refused"), false},
	}
	for _, tc := range cases {
		if got := isInputRejection(tc.err); got != tc.want {
			t.Errorf("%s: isInputRejection = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// A provider that answers "too large" is alive; the same input will be refused
// on every try. Counting those refusals tripped the only local encoder for an
// hour during a bulk merge on 2026-09-25.
func TestInputRejectionsDoNotTripTheBreaker(t *testing.T) {
	e := llamaServer(t, http.StatusInternalServerError,
		`{"error":{"code":500,"message":"input (40003 tokens) is too large to process. increase the physical batch size","type":"server_error"}}`)

	for i := 0; i < providerFailureThreshold+2; i++ {
		if _, err := e.EmbedDetailed(context.Background(), "огромная сводка"); err == nil {
			t.Fatal("an oversize input must still fail")
		}
	}
	if e.healthFor("llamacpp").isOpen() {
		t.Fatal("breaker opened on input rejections — every unrelated write would lose its vector")
	}
}

// A genuine outage still trips it, and for a local provider only briefly.
func TestLocalProviderOutageTripsForAShortCooldown(t *testing.T) {
	e := llamaServer(t, http.StatusServiceUnavailable, "loading model")

	for i := 0; i < providerFailureThreshold; i++ {
		if _, err := e.EmbedDetailed(context.Background(), "x"); err == nil {
			t.Fatal("503 must fail")
		}
	}
	h := e.healthFor("llamacpp")
	if !h.isOpen() {
		t.Fatalf("breaker must open after %d consecutive outages", providerFailureThreshold)
	}
	h.mu.Lock()
	remaining := time.Until(h.disabledUntil)
	h.mu.Unlock()
	if remaining > localProviderDisableCooldown {
		t.Errorf("local provider disabled for %v, want at most %v", remaining, localProviderDisableCooldown)
	}
}

// The exhaustion error must describe what happened. It used to tell the reader
// to configure JINA/OpenAI/Ollama while llama.cpp was configured and merely
// skipped by its breaker.
func TestExhaustedErrorNamesTheOpenBreaker(t *testing.T) {
	e := llamaServer(t, http.StatusServiceUnavailable, "loading model")
	for i := 0; i < providerFailureThreshold; i++ {
		_, _ = e.EmbedDetailed(context.Background(), "x") // intentionally ignored: driving the breaker open
	}

	_, err := e.EmbedDetailed(context.Background(), "x")
	if err == nil {
		t.Fatal("expected an error with the only provider's breaker open")
	}
	msg := err.Error()
	if !strings.Contains(msg, "circuit breaker open for llamacpp") {
		t.Errorf("error %q does not name the open breaker", msg)
	}
	if strings.Contains(msg, "JINA_API_KEY") {
		t.Errorf("error %q sends the reader to fix a configuration that is fine", msg)
	}
}

func TestExhaustedErrorListsTriedProviders(t *testing.T) {
	e := llamaServer(t, http.StatusInternalServerError, "input is too large to process")
	_, err := e.EmbedDetailed(context.Background(), "x")
	if err == nil || !strings.Contains(err.Error(), "tried: llamacpp") {
		t.Fatalf("error = %v, want it to name the provider that was tried", err)
	}
}
