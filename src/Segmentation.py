import SimpleITK as sitk
import matplotlib.pyplot as plt

def modify_segmentation(input_path, output_path):
    """
    Abre una segmentación en formato .nrrd, cambia todos los valores 2 a 0,
    y guarda la segmentación modificada.

    Parámetros:
    - input_path: Ruta del archivo .nrrd de entrada.
    - output_path: Ruta donde se guardará el archivo .nrrd modificado.
    """
    # Cargar la segmentación
    segmentation = sitk.ReadImage(input_path)
    segmentation_array = sitk.GetArrayFromImage(segmentation)

    # Cambiar los valores 2 a 0
    segmentation_array[segmentation_array == 2] = 0

    # Guardar la segmentación modificada
    modified_segmentation = sitk.GetImageFromArray(segmentation_array)
    modified_segmentation.CopyInformation(segmentation)
    sitk.WriteImage(modified_segmentation, output_path)

    print(f"Segmentación modificada guardada en: {output_path}")




modify_segmentation("\\\\10.5.38.120\\BIT-UPM-projects\\INGENIO-RAD\\DATA\\cleanData\\NRRD\\HUB\\HUB_018\\INGENIO_HUB_018_TH_CT_seg.nrrd", "..\\INGENIO_HUB_018_TH_CT_seg_v2.nrrd")