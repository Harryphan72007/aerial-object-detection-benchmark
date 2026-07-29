import numpy as np
from src.evaluation.detection_metrics import greedy_match,iou_matrix,xywh_to_xyxy
from src.evaluation.efficiency_metrics import pareto_frontier,summarize_latencies

def test_iou_and_matching():
    a=np.array([xywh_to_xyxy([0,0,10,10])]);b=np.array([xywh_to_xyxy([0,0,10,10])]);assert np.isclose(iou_matrix(a,b)[0,0],1)
    gt=[{"bbox":[0,0,10,10],"category_id":1}];pred=[{"bbox":[0,0,10,10],"category_id":1,"score":.9}]
    assert greedy_match(gt,pred)[0]["is_tp"]

def test_latency_summary():
    x=summarize_latencies([10,20,30]);assert np.isclose(x["mean_latency_ms"],20);assert np.isclose(x["fps"],50)

def test_pareto():
    rows=[{"id":"a","acc":.5,"cost":1},{"id":"b","acc":.6,"cost":2},{"id":"c","acc":.4,"cost":3}]
    assert {r["id"] for r in pareto_frontier(rows,"acc","cost")}=={"a","b"}

from src.evaluation.robustness import Tile, generate_tiles, merge_tiled_predictions


def test_tiles_cover_edges_and_merge_duplicates():
    tiles = generate_tiles(100, 80, 64, overlap=0.25)
    assert any(tile.x + tile.width == 100 for tile in tiles)
    assert any(tile.y + tile.height == 80 for tile in tiles)
    merged, latency = merge_tiled_predictions(
        [
            (Tile(0, 0, 64, 64), [{"category_id": 1, "score": 0.9, "bbox": [50, 10, 10, 10]}]),
            (Tile(40, 0, 60, 64), [{"category_id": 1, "score": 0.8, "bbox": [10, 10, 10, 10]}]),
        ],
        image_width=100,
        image_height=80,
        image_id=7,
    )
    assert len(merged) == 1
    assert merged[0]["image_id"] == 7
    assert latency >= 0
