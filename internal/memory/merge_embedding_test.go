package memory

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"unicode/utf8"
)

// A merged primary is the concatenation of every duplicate, so it is exactly
// the body the encoder refuses. MergeDuplicates called the encoder directly and
// skipped the prefix retry: on 2026-09-25 a bulk merge left 85 primaries
// without a vector and tripped the provider's breaker for an hour.
func TestMergeDuplicatesEmbedsOversizeBodyByItsOpening(t *testing.T) {
	store, emb := newSizeLimitedStore(t, embedRetryRunes)
	ctx := context.Background()

	primary := &Memory{Title: "Session close / утро", Content: strings.Repeat("разбор первой сессии. ", 150), Type: TypeEpisodic}
	dup := &Memory{Title: "Session close / вечер", Content: strings.Repeat("разбор второй сессии. ", 150), Type: TypeEpisodic}
	for _, m := range []*Memory{primary, dup} {
		if err := store.Store(ctx, m); err != nil {
			t.Fatalf("Store: %v", err)
		}
	}
	emb.calls = nil

	if _, err := store.MergeDuplicates(ctx, primary.ID, []string{dup.ID}); err != nil {
		t.Fatalf("MergeDuplicates: %v", err)
	}

	if len(emb.calls) != 2 || emb.calls[1] != embedRetryRunes {
		t.Fatalf("encoder attempts = %v, want the merged body then its %d-rune opening", emb.calls, embedRetryRunes)
	}
	merged, err := store.Get(primary.ID)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if len(merged.Embedding) == 0 {
		t.Fatal("merged primary stored without a vector — the 2026-09-25 regression")
	}
	if merged.Metadata[MetadataEmbeddingTruncated] != "true" {
		t.Errorf("metadata %s = %q, want \"true\"", MetadataEmbeddingTruncated, merged.Metadata[MetadataEmbeddingTruncated])
	}
}

// The partiality mark describes a vector. When a merge replaces that vector
// with a whole one, the mark must go; in the live bank 11 of 21 vectorless
// merged primaries still said embedding_truncated=true.
func TestMergeDuplicatesDropsStaleTruncationMark(t *testing.T) {
	store, emb := newSizeLimitedStore(t, embedRetryRunes)
	ctx := context.Background()

	primary := &Memory{Title: "длинная", Content: strings.Repeat("слово ", 1000), Type: TypeSemantic}
	dup := &Memory{Title: "короткая", Content: "другая мысль", Type: TypeSemantic}
	for _, m := range []*Memory{primary, dup} {
		if err := store.Store(ctx, m); err != nil {
			t.Fatalf("Store: %v", err)
		}
	}
	before, err := store.Get(primary.ID)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if before.Metadata[MetadataEmbeddingTruncated] != "true" {
		t.Fatalf("precondition: primary should carry a partial vector")
	}

	emb.maxRunes = 1 << 20 // the encoder's batch was raised
	if _, err := store.MergeDuplicates(ctx, primary.ID, []string{dup.ID}); err != nil {
		t.Fatalf("MergeDuplicates: %v", err)
	}

	merged, err := store.Get(primary.ID)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if _, ok := merged.Metadata[MetadataEmbeddingTruncated]; ok {
		t.Errorf("merged primary still marked %s although its vector covers the whole body", MetadataEmbeddingTruncated)
	}
}

// update_memory refuses a body above userio.MaxMemoryContentLen *bytes*. A merge
// that produces a larger one leaves a record nobody can edit through the tool —
// merged primaries reached 262 KB. Cyrillic is two bytes per rune, which is
// what the old rune budget missed.
func TestMergeContentStaysWithinTheUpdateLimit(t *testing.T) {
	// ~60K runes but ~120K bytes: under a rune budget, over the byte limit —
	// the exact gap the old check left open.
	duplicates := make([]*Memory, 0, 20)
	for i := 0; i < 20; i++ {
		duplicates = append(duplicates, &Memory{
			Title:   fmt.Sprintf("Session close / %d", i),
			Content: fmt.Sprintf("сессия %d: ", i) + strings.Repeat("ё", 3000),
		})
	}

	got := mergeContent("основная запись", duplicates)

	if len(got) > MaxMergedContentBytes {
		t.Errorf("merged content is %d bytes, limit %d", len(got), MaxMergedContentBytes)
	}
	if !utf8.ValidString(got) {
		t.Error("merged content is not valid UTF-8 — the cut split a rune")
	}
	if !strings.HasSuffix(got, mergedContentTruncatedSuffix) {
		t.Error("an over-limit merge must say it was truncated")
	}
}
