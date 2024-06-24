import os
import sys
import time
import logging
from datetime import datetime
import pydicom
import SimpleITK as sitk
import json


def process_dicom_files(input_folder, output_folder, logs_folder, config_folder):
    # Create a log file with the current date and time as filename
    log_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
    log_filepath = os.path.join(logs_folder, log_filename)

    # Create a log file with the current date and time as filename
    config_filename = "config.json"
    config_filepath = os.path.join(config_folder, config_filename)
    if os.path.exists(config_filepath):
        with open(config_filepath, 'r') as fp:
            config_js = json.load(fp)
    else:
        config_js = {'threshold1': 100,
                     'threshold2': 200,
                     }

    logging.basicConfig(filename=log_filepath, level=logging.DEBUG)

    logging.debug(f'Input folder:{input_folder}')
    logging.debug(f'Output folder:{output_folder}')
    logging.debug(f'Logs folder:{logs_folder}')
    logging.debug(f'Config folder:{config_folder}')

    logging.debug(f'List files:{os.listdir(input_folder)}')

    for filename in os.listdir(input_folder):
        dicom_path = os.path.join(input_folder, filename)

        # Read the DICOM file
        ds = pydicom.dcmread(dicom_path)

        # Extract SeriesDescription and StudyDate
        series_description = ds.SeriesDescription if 'SeriesDescription' in ds else "N/A"
        study_date = ds.StudyDate if 'StudyDate' in ds else "N/A"

        # Log the extracted information
        logging.info("LOGS INFO")
        log_entry = f"SeriesDescription: {series_description}, StudyDate: {study_date}"
        logging.debug(log_entry)

        # Get the image data from the DICOM file
        try:
            image = ds.pixel_array

            # Convert the image to a SimpleITK Image
            sitk_image = sitk.GetImageFromArray(image)
            sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)

            # Apply Canny edge detection
            edges = sitk.CannyEdgeDetection(sitk_image,
                                            lowerThreshold=config_js['threshold1'],
                                            upperThreshold=config_js['threshold2'])

            edges = sitk.Cast(edges, sitk.sitkInt16)

            # Convert the result back to a numpy array
            edges_array = sitk.GetArrayFromImage(edges)
            logging.debug(f'shape {edges_array.shape}')

            logging.debug(f'checkpoint 0')
            # Check the shape of edges_array and set Rows and Columns accordingly
            if edges_array.ndim == 2:
                rows, columns = edges_array.shape
            elif edges_array.ndim == 3:
                rows, columns, _ = edges_array.shape
            else:
                raise ValueError(f"Unexpected number of dimensions in edges_array: {edges_array.ndim}")

            # Create a new DICOM dataset for the output
            new_ds = ds.copy()
            new_ds.PixelData = edges_array.tobytes()
            logging.debug(f'checkpoint 1')
            new_ds.Rows = rows
            new_ds.Columns = columns

            # Write the new DICOM file to the output folder
            output_dicom_path = os.path.join(output_folder, filename)
            new_ds.save_as(output_dicom_path)
            logging.debug(f'checkpoint 2')
        except Exception as e:
            logging.error('Error at %s', 'division', exc_info=e)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(len(sys.argv))
        print("Usage: simpleprototype_app.exe <input_folder> <output_folder> <logs_folder> <config_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    logs_folder = sys.argv[3]
    config_folder = sys.argv[4]

    process_dicom_files(input_folder, output_folder, logs_folder, config_folder)
