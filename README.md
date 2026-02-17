AI Powered Interview Coach

The AI Interview Coach is an automated platform designed to help job seekers practice technical interviews in a realistic, high-pressure environment. It combines natural language processing and computer vision to evaluate both technical knowledge and behavioral performance.
Project Approach
The core philosophy of this project is to move beyond generic question banks. The system uses a data-driven approach to ensure every interview is tailored to the specific user and the role they are pursuing.
The project follows a three-stage logic:
1.	Vector-Based Personalization: The system extracts text from an uploaded resume and a job description. It converts these into mathematical vectors and uses Cosine Similarity to measure the alignment between the candidate's skills and the job requirements.
2.	Immersive Vocal Interaction: To simulate a real person, the system uses Text-to-Speech to ask questions. This removes the "reading" element of prep and forces users to rely on their listening skills, as they would in a real call.
3.	Behavioral Monitoring: While the user speaks, the system monitors their physical engagement. By tracking facial landmarks, it can determine if a user is maintaining eye contact or looking away to read notes.


Features Implemented:

->The application includes several specialized modules for a complete interview simulation:
->Resume Alignment Engine Using Cosine Similarity, the system identifies how closely the candidate's background matches the target role. This score is used to validate the relevance of the interview before the simulation begins.
->Vocalized Interview Questions The AI acts as a vocal interviewer. It reads questions aloud, creating a hands-free experience where the user only needs to focus on their verbal responses.
->Real-Time Focus Proctoring The system uses facial mesh technology to track the user’s gaze. If the user looks away from the screen for an extended period, the system logs this as a "distraction" and provides feedback on their professional delivery.
->Speech Performance Analytics Every answer is transcribed and analyzed. The system calculates the user's speaking pace (Beats Per Minute) and flags the use of filler words, helping users identify where they sound hesitant or rushed.
->Dynamic Performance Dashboard After the interview, the system generates a visual report. This includes a final "Confidence Score," a breakdown of technical keyword relevance, and a timeline of the user's focus throughout the session.


Technology Stack and Components
The project is built using the following verified libraries and frameworks:

User Interface
•	Streamlit: Used to build the web interface and manage the state of the interview session.

Artificial Intelligence and NLP
•	OpenAI Whisper: Used for high-accuracy speech-to-text transcription of the user's answers.
•	Sentence Transformers: Used to create the text embeddings required for similarity matching.
•	Scikit-Learn: Specifically the Cosine Similarity module for calculating resume-to-job-description alignment.
•	gTTS (Google Text-to-Speech): Provides the vocal engine for the AI interviewer.

Computer Vision
•	MediaPipe: Handles real-time facial landmark detection to track the user's head pose and gaze.
•	OpenCV: Manages the video stream processing and displays the proctoring status on the screen.

Data and Audio Processing
•	Librosa: Analyzes audio files to extract rhythm and tempo for speech pacing metrics.
•	PDFPlumber: Extracts clean text from PDF resumes for the initial matching process.

