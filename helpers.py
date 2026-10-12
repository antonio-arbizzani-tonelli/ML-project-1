"""CSV submission helper adapted from the EPFL Project 1 starter code.

Source: https://github.com/epfml/ML_course/blob/main/projects/project1/helpers.py
UTF-8 and LF line endings preserve the bytes of our verified submission.
"""

import csv


def create_csv_submission(ids, y_pred, name):
    """Write Id,Prediction rows with signed labels in the supplied ID order."""
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")
    with open(name, "w", encoding="utf-8", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(
            csvfile, delimiter=",", fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})
