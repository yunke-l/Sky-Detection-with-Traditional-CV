# Sky-Detection-with-Traditional-CV
NEU CS5330 Course Lab Project 1: A computer vision application to identify sky pixels in a set of images using traditional image processing techniques with OpenCV. 

## Gradio DEMO
https://huggingface.co/spaces/RedBottle13/Sky-Segmentation


## Implementation
### Approach
- Step 1: Transform BGR images into HSV format.
- Step 2: Determine the HSV range using a dataset of sky images.
- Step 3: Employ the identified HSV range to create an initial mask.
- Step 4: Execute morphological operations for fine-tuning the mask.
- Step 5: Conduct connected component analysis to complete the sky mask.

### Assumptions
1. The sky is consistently located in the upper portion of the images.
2. Sky images fall into three categories based on conditions:
- Clear/Sunny: Displaying bright blue color.
- Sunrise/Sunset: Displaying red, orange, or pink colors.
- Overcast: Displaying pale or gray colors.
3. The sky region is a contiguous area.

### Sky Image Datasets
1. Sky Recognition Dataset
- Purpose: Analyzes the sky during daylight and sunset.
- Content: Consists of 58 ground-based sky images.
- Source: [Sky Recognition Dataset](https://www.kaggle.com/datasets/dinarakhudjatova/skydaylightrecognitiondataset)

2. Skylines 12 Dataset
- Purpose: Provides a diverse range of high-resolution images.
- Content: Includes 120 images capturing skylines from 12 different cities.
- Source: [Skylines 12 Dataset](https://www.kaggle.com/datasets/vassiliskrikonis/skylines-12)

3. New York City Midtown with Empire State Building at Sunset Stock Photo
- Purpose: Supplements the dataset with a specific urban sunset scene.
- Source: [New York City Midtown with Empire State Building at Sunset Stock Photo](https://www.istockphoto.com/photo/new-york-city-midtown-with-empire-state-building-at-sunset-gm521714583-50356054)

## Sample Results
