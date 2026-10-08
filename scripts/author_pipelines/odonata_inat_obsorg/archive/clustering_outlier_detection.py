import critterframe as cf
import logging
logging.basicConfig(level=logging.INFO)

from critterframe.extensions.bioencoder.embedding import load_registered as load_bioencoder
from critterframe.metrics.embedding import pretrained

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

# off-the-shelf: an ImageNet resnet18 from timm, classifier removed
# resize="pad" so a long, thin abdomen isn't squashed into a square.
resnet_embedding = cf.embedding(pretrained("resnet18", resize="pad"),
                                name="resnet18_embedding")
cf.run_metrics(
    PROJECT_PATH,
    run_name="resnet18_embedding_organism",
    transforms=[cf.remove_background(), cf.crop_to_mask(), cf.orient(axis_strategy="longer")],
    metrics=[cf.embedding(pretrained("resnet18", resize="pad"),
                                name="resnet18_embedding")],   # run_name defaults to "resnet18_embedding"
    visualize_every=250,
)