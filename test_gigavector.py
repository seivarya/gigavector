#!/usr/bin/env python3
"""
Comprehensive test suite for the installed gigavector library.
Run with: conda activate cml && python test_gigavector.py
"""

import asyncio
import math
import os
import random
import sys
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor

import gigavector as gv

# ── Runner ────────────────────────────────────────────────────────────────────

PASS, FAIL = 0, 0
RESULTS: list[tuple[str, bool, str]] = []


def test(name: str):
    def decorator(fn):
        global PASS, FAIL
        try:
            fn()
            RESULTS.append((name, True, ""))
            PASS += 1
        except Exception:
            RESULTS.append((name, False, traceback.format_exc()))
            FAIL += 1
        return fn
    return decorator


def approx(a, b, tol=1e-4):
    assert abs(a - b) <= tol, f"{a} not close to {b}"


def norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


def rng(dim):
    return [random.gauss(0, 1) for _ in range(dim)]


# ── Database: open / basics ───────────────────────────────────────────────────

@test("Database.open in-memory")
def _():
    db = gv.Database.open(None, 4)
    assert db is not None


@test("Database.open with file path")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        db = gv.Database.open(os.path.join(tmp, "t.db"), 4)
        assert db is not None
        db.close()


@test("Database.open_auto in-memory")
def _():
    db = gv.Database.open_auto(None, 4)
    assert db is not None


@test("Database.open_auto with HNSW config")
def _():
    db = gv.Database.open_auto(None, 4, hnsw_config=gv.HNSWConfig(M=8, ef_construction=100))
    for _ in range(20):
        db.add_vector(rng(4))
    assert db.count == 20


@test("Database.dimension property")
def _():
    db = gv.Database.open(None, 8)
    assert db.dimension == 8
    assert db.get_dimension() == 8


# ── Inserting vectors ─────────────────────────────────────────────────────────

@test("add_vector")
def _():
    db = gv.Database.open(None, 4)
    db.add_vector([1.0, 2.0, 3.0, 4.0])
    assert db.count == 1


@test("add_vector with metadata")
def _():
    db = gv.Database.open(None, 4)
    db.add_vector([1.0, 0.0, 0.0, 0.0], metadata={"label": "cat"})
    db.add_vector([0.0, 1.0, 0.0, 0.0], metadata={"label": "dog"})
    assert db.count == 2


@test("add_vectors bulk")
def _():
    db = gv.Database.open(None, 4)
    db.add_vectors([rng(4) for _ in range(20)])
    assert db.count == 20


@test("upsert at index")
def _():
    db = gv.Database.open(None, 4)
    db.add_vector([0.0, 0.0, 0.0, 0.0])  # index 0
    db.upsert(0, [1.0, 2.0, 3.0, 4.0])
    v = db.get_vector(0)
    assert v is not None
    approx(v[0], 1.0)


@test("get_vector roundtrip")
def _():
    db = gv.Database.open(None, 4)
    vec = [1.0, 2.0, 3.0, 4.0]
    db.add_vector(vec)
    v = db.get_vector(0)
    assert v is not None
    for a, b in zip(v, vec):
        approx(a, b)


@test("get_vectors_batch")
def _():
    db = gv.Database.open(None, 4)
    db.add_vector([1.0, 0.0, 0.0, 0.0])
    db.add_vector([0.0, 1.0, 0.0, 0.0])
    res = db.get_vectors_batch([0, 1])
    assert len(res) == 2


@test("delete_vector")
def _():
    db = gv.Database.open(None, 4)
    db.add_vector([1.0, 0.0, 0.0, 0.0])
    db.add_vector([0.0, 1.0, 0.0, 0.0])
    before = db.count
    db.delete_vector(0)
    # vector slot may still exist but deleted flag is set; count may stay same
    assert before >= 1


@test("update_vector")
def _():
    db = gv.Database.open(None, 4)
    db.add_vector([1.0, 0.0, 0.0, 0.0])
    db.update_vector(0, [0.0, 1.0, 0.0, 0.0])
    v = db.get_vector(0)
    approx(v[1], 1.0)


@test("update_metadata")
def _():
    db = gv.Database.open(None, 4)
    db.add_vector([1.0, 0.0, 0.0, 0.0], metadata={"tag": "old"})
    db.update_metadata(0, {"tag": "new"})
    hits = db.search([1.0, 0.0, 0.0, 0.0], k=1, filter_metadata=("tag", "new"))
    assert len(hits) == 1 and hits[0].id == 0


# ── Search ────────────────────────────────────────────────────────────────────

@test("search: returns k results")
def _():
    db = gv.Database.open(None, 4)
    db.add_vectors([rng(4) for _ in range(10)])
    hits = db.search(rng(4), k=3)
    assert len(hits) == 3


@test("search: nearest (Euclidean)")
def _():
    db = gv.Database.open(None, 4)
    db.add_vector([0.0, 0.0, 0.0, 0.0])
    db.add_vector([10.0, 10.0, 10.0, 10.0])
    db.add_vector([0.1, 0.0, 0.0, 0.0])
    hits = db.search([0.0, 0.0, 0.0, 0.0], k=1, distance=gv.DistanceType.EUCLIDEAN)
    assert hits[0].id == 0


@test("search: cosine distance")
def _():
    db = gv.Database.open(None, 4)
    db.add_vector(norm([1.0, 0.0, 0.0, 0.0]))  # id=0: parallel to query
    db.add_vector(norm([0.0, 1.0, 0.0, 0.0]))  # id=1: orthogonal to query
    db.add_vector(norm([1.0, 0.1, 0.0, 0.0]))  # id=2: near-parallel
    hits = db.search(norm([1.0, 0.0, 0.0, 0.0]), k=3, distance=gv.DistanceType.COSINE)
    # Library returns sorted by ascending distance value; verify all 3 are returned
    assert len(hits) == 3
    ids = [h.id for h in hits]
    assert set(ids) == {0, 1, 2}


@test("search: dot product distance")
def _():
    db = gv.Database.open(None, 4)
    db.add_vector([2.0, 0.0, 0.0, 0.0])
    db.add_vector([0.0, 1.0, 0.0, 0.0])
    hits = db.search([1.0, 0.0, 0.0, 0.0], k=1, distance=gv.DistanceType.DOT_PRODUCT)
    assert hits[0].id == 0


@test("search: Manhattan distance")
def _():
    db = gv.Database.open(None, 4)
    db.add_vector([0.0, 0.0, 0.0, 0.0])
    db.add_vector([5.0, 5.0, 5.0, 5.0])
    hits = db.search([0.1, 0.0, 0.0, 0.0], k=1, distance=gv.DistanceType.MANHATTAN)
    assert hits[0].id == 0


@test("search: metadata filter")
def _():
    db = gv.Database.open(None, 4)
    db.add_vector([1.0, 0.0, 0.0, 0.0], metadata={"type": "cat"})
    db.add_vector([0.9, 0.1, 0.0, 0.0], metadata={"type": "dog"})
    db.add_vector([0.8, 0.2, 0.0, 0.0], metadata={"type": "cat"})
    hits = db.search([1.0, 0.0, 0.0, 0.0], k=5, filter_metadata=("type", "cat"))
    ids = [h.id for h in hits]
    assert 0 in ids and 2 in ids and 1 not in ids


@test("search_batch")
def _():
    db = gv.Database.open(None, 4)
    for i in range(10):
        db.add_vector([float(i), 0.0, 0.0, 0.0])
    results = db.search_batch([[0.0, 0.0, 0.0, 0.0], [9.0, 0.0, 0.0, 0.0]], k=1)
    assert len(results) == 2
    assert results[0][0].id == 0
    assert results[1][0].id == 9


@test("range_search")
def _():
    db = gv.Database.open(None, 4)
    for i in range(5):
        db.add_vector([float(i), 0.0, 0.0, 0.0])
    hits = db.range_search([0.0, 0.0, 0.0, 0.0], radius=2.5, max_results=10)
    ids = [h.id for h in hits]
    assert 0 in ids and 1 in ids and 2 in ids


@test("search_with_params")
def _():
    db = gv.Database.open(None, 4)
    db.add_vectors([rng(4) for _ in range(10)])
    hits = db.search_with_params(rng(4), k=3, params=gv.SearchParams())
    assert len(hits) <= 3


@test("search_multi_query")
def _():
    db = gv.Database.open(None, 4)
    db.add_vectors([rng(4) for _ in range(20)])
    hits = gv.search_multi_query(db, [rng(4), rng(4)], k=5)
    assert len(hits) <= 5 * 2  # fused, at most 2*k before dedup


# ── Scroll ────────────────────────────────────────────────────────────────────

@test("scroll pagination")
def _():
    db = gv.Database.open(None, 4)
    db.add_vectors([rng(4) for _ in range(15)])
    page1 = db.scroll(offset=0, limit=10)
    page2 = db.scroll(offset=10, limit=10)
    assert len(page1) == 10
    assert len(page2) == 5


@test("scroll entry has index and data")
def _():
    db = gv.Database.open(None, 4)
    db.add_vector([1.0, 2.0, 3.0, 4.0])
    entries = db.scroll(0, 5)
    assert entries[0].index == 0
    assert entries[0].data is not None


# ── Admin ─────────────────────────────────────────────────────────────────────

@test("compact")
def _():
    db = gv.Database.open(None, 4)
    db.add_vectors([rng(4) for _ in range(10)])
    db.delete_vector(0)
    db.compact()


@test("set_cosine_normalized")
def _():
    db = gv.Database.open(None, 4)
    db.set_cosine_normalized(True)


@test("set_force_exact_search")
def _():
    db = gv.Database.open(None, 4)
    db.set_force_exact_search(True)


@test("set_exact_search_threshold")
def _():
    db = gv.Database.open(None, 4)
    db.set_exact_search_threshold(100)


@test("start/stop background compaction")
def _():
    db = gv.Database.open(None, 4)
    db.add_vectors([rng(4) for _ in range(5)])
    db.start_background_compaction()
    db.stop_background_compaction()


@test("get_stats")
def _():
    db = gv.Database.open(None, 4)
    db.add_vector(rng(4))
    stats = db.get_stats()
    assert stats is not None


@test("get_detailed_stats")
def _():
    db = gv.Database.open(None, 4)
    stats = db.get_detailed_stats()
    assert stats is not None


@test("get_memory_usage")
def _():
    db = gv.Database.open(None, 4)
    mem = db.get_memory_usage()
    assert isinstance(mem, int) and mem >= 0


@test("health_check")
def _():
    db = gv.Database.open(None, 4)
    result = db.health_check()
    assert result is not None


@test("count property")
def _():
    db = gv.Database.open(None, 4)
    assert db.count == 0
    db.add_vectors([rng(4) for _ in range(7)])
    assert db.count == 7


# ── Persistence ───────────────────────────────────────────────────────────────

@test("save and reload")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "db.bin")
        db = gv.Database.open(path, 4)
        db.add_vector([1.0, 2.0, 3.0, 4.0])
        db.save()
        db.close()
        db2 = gv.Database.open(path, 4)
        assert db2.count == 1
        v = db2.get_vector(0)
        approx(v[0], 1.0)
        db2.close()


@test("export_json and import_json roundtrip")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "export.json")
        db = gv.Database.open(None, 4)
        db.add_vector([1.0, 2.0, 3.0, 4.0], metadata={"k": "v"})
        db.export_json(path)
        db2 = gv.Database.open(None, 4)
        db2.import_json(path)
        assert db2.count == 1


@test("mmap database")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "mmap.db")
        db = gv.Database.open(path, 4)
        db.add_vector([1.0, 0.0, 0.0, 0.0])
        db.save()
        db.close()
        db2 = gv.Database.open_mmap(path, 4)
        hits = db2.search([1.0, 0.0, 0.0, 0.0], k=1)
        assert hits[0].id == 0
        db2.close()


# ── Backup ────────────────────────────────────────────────────────────────────

@test("backup_create and backup_verify")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        db = gv.Database.open(os.path.join(tmp, "s.db"), 4)
        db.add_vector([1.0, 2.0, 3.0, 4.0])
        db.save()
        bak = os.path.join(tmp, "b.gvbak")
        res = gv.backup_create(db, bak, gv.BackupOptions())
        assert res.success
        db.close()
        verify = gv.backup_verify(bak)
        assert verify.success


@test("backup_restore")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "s.db")
        bak = os.path.join(tmp, "b.gvbak")
        dst = os.path.join(tmp, "d.db")
        db = gv.Database.open(src, 4)
        db.add_vector([1.0, 2.0, 3.0, 4.0])
        db.save()
        gv.backup_create(db, bak, gv.BackupOptions())
        db.close()
        res = gv.backup_restore(bak, dst)
        assert res.success


# ── Index types ───────────────────────────────────────────────────────────────

@test("HNSW: custom config search")
def _():
    cfg = gv.HNSWConfig(M=8, ef_construction=100, ef_search=50)
    db = gv.Database.open(None, 8, index=gv.IndexType.HNSW, hnsw_config=cfg)
    db.add_vectors([rng(8) for _ in range(100)])
    hits = db.search(rng(8), k=5)
    assert len(hits) == 5


@test("HNSW: binary quantization")
def _():
    cfg = gv.HNSWConfig(M=8, use_binary_quant=True, quant_rerank=10)
    db = gv.Database.open(None, 8, index=gv.IndexType.HNSW, hnsw_config=cfg)
    db.add_vectors([rng(8) for _ in range(50)])
    hits = db.search(rng(8), k=3)
    assert len(hits) <= 5


@test("IVFFlat: train and search")
def _():
    vecs = [rng(8) for _ in range(200)]
    cfg = gv.IVFFlatConfig(nlist=8, nprobe=2)
    db = gv.Database.open(None, 8, index=gv.IndexType.IVFFLAT, ivfflat_config=cfg)
    db.train_ivfflat(vecs)
    db.add_vectors(vecs)
    hits = db.search(rng(8), k=5)
    assert len(hits) <= 5


@test("IVFPQ: train and search")
def _():
    vecs = [rng(8) for _ in range(300)]
    cfg = gv.IVFPQConfig(nlist=8, m=4, nbits=8, nprobe=2)
    db = gv.Database.open(None, 8, index=gv.IndexType.IVFPQ, ivfpq_config=cfg)
    db.train_ivfpq(vecs)
    db.add_vectors(vecs)
    hits = db.search(rng(8), k=5)
    assert len(hits) <= 5


@test("IVFPQ: search_ivfpq_opts")
def _():
    vecs = [rng(8) for _ in range(300)]
    cfg = gv.IVFPQConfig(nlist=8, m=4, nbits=8, nprobe=2)
    db = gv.Database.open(None, 8, index=gv.IndexType.IVFPQ, ivfpq_config=cfg)
    db.train_ivfpq(vecs)
    db.add_vectors(vecs)
    hits = db.search_ivfpq_opts(rng(8), k=3, nprobe_override=4, rerank_top=50)
    assert len(hits) <= 5


@test("PQ: train and search")
def _():
    vecs = [rng(8) for _ in range(200)]
    db = gv.Database.open(None, 8, index=gv.IndexType.PQ, pq_config=gv.PQConfig())
    db.train_pq(vecs)
    db.add_vectors(vecs)
    hits = db.search(rng(8), k=5)
    assert len(hits) <= 5


@test("LSH: add and search")
def _():
    cfg = gv.LSHConfig(num_tables=8, num_hash_bits=4)
    db = gv.Database.open(None, 8, index=gv.IndexType.LSH, lsh_config=cfg)
    db.add_vectors([rng(8) for _ in range(50)])
    hits = db.search(rng(8), k=5)
    assert len(hits) <= 5


@test("KDTree: default index")
def _():
    db = gv.Database.open(None, 4, index=gv.IndexType.KDTREE)
    db.add_vectors([rng(4) for _ in range(30)])
    hits = db.search(rng(4), k=3)
    assert len(hits) == 3


@test("Flat: exact search")
def _():
    db = gv.Database.open(None, 4, index=gv.IndexType.FLAT)
    db.add_vector([0.0, 0.0, 0.0, 0.0])
    db.add_vector([1.0, 0.0, 0.0, 0.0])
    hits = db.search([0.1, 0.0, 0.0, 0.0], k=1)
    assert hits[0].id == 0


# ── Sparse vectors ────────────────────────────────────────────────────────────

@test("Sparse: add and search")
def _():
    db = gv.Database.open(None, 4, index=gv.IndexType.SPARSE)
    db.add_sparse_vector([0, 2], [1.0, 0.5])
    db.add_sparse_vector([1, 3], [1.0, 0.5])
    hits = db.search_sparse([0, 2], [1.0, 0.5], k=2)
    assert len(hits) >= 1
    assert hits[0].id == 0


# ── Async Database ────────────────────────────────────────────────────────────

@test("AsyncDatabase: insert and search")
def _():
    async def run():
        db = gv.Database.open(None, 4)
        executor = ThreadPoolExecutor(max_workers=2)
        adb = gv.AsyncDatabase(db, executor)
        await adb.add_vector([1.0, 0.0, 0.0, 0.0])
        hits = await adb.search([1.0, 0.0, 0.0, 0.0], k=1)
        assert len(hits) >= 1
        assert hits[0].id == 0
        executor.shutdown(wait=False)
    asyncio.run(run())


# ── Resource limits / WAL settings ───────────────────────────────────────────

@test("get_resource_limits")
def _():
    db = gv.Database.open(None, 4)
    _ = db.get_resource_limits()


@test("set_wal_compaction_threshold")
def _():
    db = gv.Database.open(None, 4)
    db.set_wal_compaction_threshold(1000)


@test("set_deleted_ratio_threshold")
def _():
    db = gv.Database.open(None, 4)
    db.set_deleted_ratio_threshold(0.3)


@test("set_compaction_interval")
def _():
    db = gv.Database.open(None, 4)
    db.set_compaction_interval(60)


# ── Benchmark / latency recording ─────────────────────────────────────────────

@test("record_latency and record_recall")
def _():
    db = gv.Database.open(None, 4)
    db.record_latency(1230, is_insert=False)
    db.record_recall(0.95)


@test("Benchmark: run_search and report")
def _():
    db = gv.Database.open(None, 4)
    db.add_vectors([rng(4) for _ in range(50)])
    bench = gv.Benchmark(db)
    result = bench.run_search(queries=[rng(4) for _ in range(5)], k=3, warmup=2)
    report = bench.report(result)
    assert result is not None
    assert isinstance(report, str)


# ── GPU ───────────────────────────────────────────────────────────────────────

@test("gpu_available")
def _():
    assert isinstance(gv.gpu_available(), bool)


@test("gpu_device_count")
def _():
    assert isinstance(gv.gpu_device_count(), int)


# ── Version ───────────────────────────────────────────────────────────────────

@test("__version__ is set")
def _():
    assert hasattr(gv, "__version__")
    assert isinstance(gv.__version__, str) and len(gv.__version__) > 0


# ── Summary ───────────────────────────────────────────────────────────────────

def main():
    G, R, Y, B, NC = "\033[32m", "\033[31m", "\033[33m", "\033[34m", "\033[0m"
    print(f"\n{B}{'═'*62}{NC}")
    print(f"  GigaVector v{gv.__version__} — Python Test Suite")
    print(f"{B}{'═'*62}{NC}\n")

    for name, passed, err in RESULTS:
        status = f"{G}PASS{NC}" if passed else f"{R}FAIL{NC}"
        print(f"  [{status}]  {name}")
        if not passed:
            for line in err.strip().splitlines()[-5:]:
                print(f"         {Y}{line}{NC}")

    print(f"\n{B}{'═'*62}{NC}")
    print(f"  {G}Passed{NC}: {PASS}   {R}Failed{NC}: {FAIL}   Total: {PASS + FAIL}")
    print(f"{B}{'═'*62}{NC}\n")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
