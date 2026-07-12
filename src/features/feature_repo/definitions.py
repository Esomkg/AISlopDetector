"""Feast feature definitions for AISlopDetector.

Defines the image embedding feature view used for drift detection
and online serving of embeddings.
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, String, Int64


image = Entity(
    name="image_id",
    description="Unique identifier for each image (filename or hash)",
    value_type=String,
)

embedding_source = FileSource(
    path="data/embeddings/image_embeddings.parquet",
    timestamp_field="collection_timestamp",
)

embedding_fields = [
    Field(name=f"embedding_{i}", dtype=Float32)
    for i in range(512)
]

image_embeddings = FeatureView(
    name="image_embeddings",
    description="CLIP ViT-B/32 embeddings (512-dim) for AISlopDetector images",
    entities=[image],
    ttl=timedelta(days=365),
    schema=embedding_fields + [
        Field(name="label", dtype=Float32),
        Field(name="generator", dtype=String),
        Field(name="collection_date", dtype=String),
    ],
    source=embedding_source,
    online=True,
    tags={"model": "clip-vit-b32", "use": "drift_detection"},
)


if __name__ == "__main__":
    print("Feast feature definitions:")
    print(f"  Entity: {image.name}")
    print(f"  Feature view: {image_embeddings.name}")
    print(f"  Fields: {len(embedding_fields)} embedding dims + label + generator + collection_date")
    print("  Run 'feast apply' from the feature_repo directory to register.")
