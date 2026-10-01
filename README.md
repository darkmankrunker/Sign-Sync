SignSync (American sign language )

SignSync is a real-time American Sign Language (ASL) alphabet
recognition system developed as a group project.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

 Features;

- Real-time hand gesture detection
- ASL alphabet recognition
- Custom-built dataset
- CNN-based classification
- MediaPipe hand tracking
- OpenCV webcam integration
- Text-to-speech output
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Technologies;

- Python
- TensorFlow / Keras
- MediaPipe
- OpenCV
- NumPy
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 Team;
  Developed as a AI group capstone project by:
- Gabriel Pinto
- Shubhang chaturvedi
- Mohammed Muhsin
- Zainsha shanavas
- Nakul Krishna
  
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

 How It Works;

The webcam captures the user's hand movements. MediaPipe detects the hand landmarks, which are converted into numerical features. The trained machine learning model then predicts the corresponding ASL alphabet letter.

Users can build words using the recognized letters and use the text-to-speech feature to hear the resulting word.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 Dataset;

The model was trained using a custom-built ASL alphabet dataset created by the project team.

The raw dataset is not included in this repository.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Running the Project;

Install the required dependencies:

```bash
pip install -r requirements.txt
