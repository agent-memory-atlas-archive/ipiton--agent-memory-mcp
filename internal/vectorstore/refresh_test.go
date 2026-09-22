package vectorstore

import (
	"path/filepath"
	"testing"

	"go.uber.org/zap"
)

// The defect these tests pin down: search answers from maps built when the
// store was opened, while the database is written by a separate process (the
// indexing run started by the post-commit hook). A reader that never reloads
// keeps answering from its startup snapshot — the commit is in vectors.db and
// missing from the results, with nothing to say so.

// openSecondReader opens another store over the same file, standing in for the
// long-lived process that did not do the indexing.
func openSecondReader(tb testing.TB, dbPath string) *SQLiteStore {
	tb.Helper()
	store, err := NewSQLiteStore(dbPath, 3, zap.NewNop())
	if err != nil {
		tb.Fatalf("NewSQLiteStore (second reader): %v", err)
	}
	tb.Cleanup(func() { _ = store.Close() })
	return store
}

func TestRefreshIfStalePicksUpAnotherProcessIndexing(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "shared-vec.db")

	writer, err := NewSQLiteStore(dbPath, 3, zap.NewNop())
	if err != nil {
		t.Fatalf("NewSQLiteStore (writer): %v", err)
	}
	defer func() { _ = writer.Close() }()

	if err := writer.Upsert([]Chunk{
		{ID: "c1", DocPath: "old.md", Content: "morning walk", Embedding: []float32{1, 0, 0}},
	}); err != nil {
		t.Fatalf("Upsert initial: %v", err)
	}
	if err := writer.SetMetadata(indexVersionKey, "2026-09-10T07:00:00Z"); err != nil {
		t.Fatalf("SetMetadata initial: %v", err)
	}

	reader := openSecondReader(t, dbPath)
	if got := reader.Count(); got != 1 {
		t.Fatalf("reader should open on 1 chunk, got %d", got)
	}

	// The other process indexes a fresh commit and stamps the version.
	if err := writer.Upsert([]Chunk{
		{ID: "c2", DocPath: "new.md", Content: "dbt-test root cause", Embedding: []float32{0, 1, 0}},
	}); err != nil {
		t.Fatalf("Upsert after commit: %v", err)
	}
	if err := writer.SetMetadata(indexVersionKey, "2026-09-10T09:30:00Z"); err != nil {
		t.Fatalf("SetMetadata after commit: %v", err)
	}

	reloaded, err := reader.RefreshIfStale()
	if err != nil {
		t.Fatalf("RefreshIfStale: %v", err)
	}
	if !reloaded {
		t.Fatal("expected a reload after the version moved")
	}
	if got := reader.Count(); got != 2 {
		t.Fatalf("expected the reader to see 2 chunks after reload, got %d", got)
	}

	results, err := reader.Search([]float32{0, 1, 0}, 5)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	var found bool
	for _, r := range results {
		if r.ID == "c2" {
			found = true
		}
	}
	if !found {
		t.Fatal("the chunk indexed by the other process is missing from search results")
	}
}

func TestRefreshIfStaleIsNoOpWhenVersionUnchanged(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "shared-vec.db")

	writer, err := NewSQLiteStore(dbPath, 3, zap.NewNop())
	if err != nil {
		t.Fatalf("NewSQLiteStore (writer): %v", err)
	}
	defer func() { _ = writer.Close() }()

	if err := writer.Upsert([]Chunk{
		{ID: "c1", DocPath: "doc.md", Content: "unchanged", Embedding: []float32{1, 0, 0}},
	}); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	if err := writer.SetMetadata(indexVersionKey, "2026-09-10T07:00:00Z"); err != nil {
		t.Fatalf("SetMetadata: %v", err)
	}

	reader := openSecondReader(t, dbPath)

	reloaded, err := reader.RefreshIfStale()
	if err != nil {
		t.Fatalf("RefreshIfStale: %v", err)
	}
	if reloaded {
		t.Fatal("expected no reload while the index version is unchanged")
	}
}

// An index that was never stamped must not read as "changed" on every call:
// that would reload the whole corpus per query.
func TestRefreshIfStaleWithoutVersionStampDoesNotReloadRepeatedly(t *testing.T) {
	store := newTestStore(t)

	if err := store.Upsert([]Chunk{
		{ID: "c1", DocPath: "doc.md", Content: "no stamp", Embedding: []float32{1, 0, 0}},
	}); err != nil {
		t.Fatalf("Upsert: %v", err)
	}

	for i := range 3 {
		reloaded, err := store.RefreshIfStale()
		if err != nil {
			t.Fatalf("RefreshIfStale #%d: %v", i, err)
		}
		if reloaded {
			t.Fatalf("call #%d reloaded an index that was never re-stamped", i)
		}
	}
}

// Concurrent searches must not each rebuild the same maps, and must all end up
// on the new copy.
func TestRefreshIfStaleUnderConcurrentReaders(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "shared-vec.db")

	writer, err := NewSQLiteStore(dbPath, 3, zap.NewNop())
	if err != nil {
		t.Fatalf("NewSQLiteStore (writer): %v", err)
	}
	defer func() { _ = writer.Close() }()

	if err := writer.Upsert([]Chunk{
		{ID: "c1", DocPath: "old.md", Content: "before", Embedding: []float32{1, 0, 0}},
	}); err != nil {
		t.Fatalf("Upsert initial: %v", err)
	}
	if err := writer.SetMetadata(indexVersionKey, "2026-09-10T07:00:00Z"); err != nil {
		t.Fatalf("SetMetadata initial: %v", err)
	}

	reader := openSecondReader(t, dbPath)

	if err := writer.Upsert([]Chunk{
		{ID: "c2", DocPath: "new.md", Content: "after", Embedding: []float32{0, 1, 0}},
	}); err != nil {
		t.Fatalf("Upsert after commit: %v", err)
	}
	if err := writer.SetMetadata(indexVersionKey, "2026-09-10T09:30:00Z"); err != nil {
		t.Fatalf("SetMetadata after commit: %v", err)
	}

	const readers = 8
	errs := make(chan error, readers)
	for range readers {
		go func() {
			_, err := reader.RefreshIfStale()
			errs <- err
		}()
	}
	for range readers {
		if err := <-errs; err != nil {
			t.Fatalf("concurrent RefreshIfStale: %v", err)
		}
	}

	if got := reader.Count(); got != 2 {
		t.Fatalf("expected 2 chunks after concurrent refresh, got %d", got)
	}
}
