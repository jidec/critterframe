from critterframe.project import paths
from critterframe.records.models import list_models, register_model

PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"
PREV_PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg_old"

src = list_models(PREV_PROJECT_PATH)["head_segmenter_v1"]
ckpt_in_a = paths.resolve_in_project(PREV_PROJECT_PATH, src[
    "path"])  # absolute Path to A's file

register_model(
    PROJECT_PATH, "head_segmenter_v1", path=ckpt_in_a,
    task=src["task"], framework=src["framework"], base_model=src["base_model"],
    parameters=src["parameters"],
    training_data=(src.get("training_data") or {}).get("dataset"),
)

src = list_models(PREV_PROJECT_PATH)["thorax_segmenter_v1"]
ckpt_in_a = paths.resolve_in_project(PREV_PROJECT_PATH, src[
    "path"])  # absolute Path to A's file

register_model(
    PROJECT_PATH, "thorax_segmenter_v1", path=ckpt_in_a,
    task=src["task"], framework=src["framework"], base_model=src["base_model"],
    parameters=src["parameters"],
    training_data=(src.get("training_data") or {}).get("dataset"),
)

src = list_models(PREV_PROJECT_PATH)["abdomen_segmenter_v1"]
ckpt_in_a = paths.resolve_in_project(PREV_PROJECT_PATH, src[
    "path"])  # absolute Path to A's file

register_model(
    PROJECT_PATH, "abdomen_segmenter_v1", path=ckpt_in_a,
    task=src["task"], framework=src["framework"], base_model=src["base_model"],
    parameters=src["parameters"],
    training_data=(src.get("training_data") or {}).get("dataset"),
)