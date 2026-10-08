import logging

import critterframe as cf

OLD_PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg_old"
NEW_PROJECT_PATH = "D:/cf_projects/odonata_inat_obsorg"

#cf.export_masks(OLD_PROJECT_PATH, dest="D:/cf_projects/masks_exports",reference=True)

cf.import_masks(NEW_PROJECT_PATH,folder="D:/cf_projects/masks_exports",reference=True)

cf.define_subset(NEW_PROJECT_PATH,"old_but_valid_parts_reference_set", occurrence_ids=sorted(set(cf.load_masks(NEW_PROJECT_PATH, parts=["head","thorax","abdomen"],reference=True, columns=["occurrence_id"])["occurrence_id"])))