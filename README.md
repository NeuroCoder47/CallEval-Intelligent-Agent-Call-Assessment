
# CallEval: Intelligent Agent Call Assessment

![CallEval Banner](https://via.placeholder.com/1200x300.png?text=CallEval+Banner) <!-- Replace with your project banner -->

## 📖 Description

**CallEval** is an innovative web application designed to assess agent calls intelligently. Leveraging advanced audio transcription and natural language processing, CallEval helps businesses evaluate their agent's performance by answering crucial questions based on the conversation. This project utilizes the Whisper model for transcription and the T5 model for analysis, providing insightful feedback on agent interactions. By automating the evaluation process, CallEval enhances productivity and facilitates training for agents, ensuring better customer service experiences.

## 📈 Objectives

- **Enhance Performance**: Provide agents with feedback to improve their performance based on recorded calls.
- **Streamline Evaluation**: Automate the analysis process to save time for managers and trainers.
- **Data-Driven Insights**: Utilize AI to gather actionable insights from call transcriptions.
- **User-Friendly Interface**: Ensure ease of use with a simple, intuitive web interface.

## 🚀 Features

- **Audio Upload**: Upload call recordings in MP3 or M4A format effortlessly.
- **WAV Conversion**: Convert audio files to WAV format using FFmpeg for better compatibility with the transcription model.
- **Automatic Transcription**: Transcribe audio files into text using the Whisper model, ensuring high accuracy.
- **Conversation Analysis**: Analyze the transcription to evaluate agent performance based on predefined criteria, such as policy adherence and effectiveness.
- **Downloadable Outputs**: Easily download converted WAV files, view transcriptions, and read analysis results for further examination.
- **Session Management**: Utilize Streamlit's session state to keep track of user inputs and results across different interactions.
- **Customizable Evaluation Criteria**: Modify the analysis questions based on specific business needs or training requirements.

## 💻 Installation

### Prerequisites

Before getting started, ensure you have the following:

- **Python 3.7+**: You can download Python from [python.org](https://www.python.org/downloads/).
- **FFmpeg**: Install FFmpeg and ensure it is added to your system PATH. You can follow the installation guide on the [FFmpeg website](https://ffmpeg.org/download.html).
- **Required Python libraries**: The necessary libraries are included in the `requirements.txt` file.

### Clone the Repository

To clone the repository, open your terminal or command prompt and run the following commands:

```bash
git clone https://github.com/yourusername/calleval.git
cd calleval
```

### Install Dependencies

Install the required Python libraries by running:

```bash
pip install -r requirements.txt
```

### Set Up Environment Variables

If necessary, set up environment variables for any sensitive information (like API keys). This can be done by creating a `.env` file in the project root.

## 🛠 Usage

1. **Run the Streamlit app**:
    Open your terminal, navigate to the project directory, and run:
    ```bash
    streamlit run app.py
    ```

2. **Upload the Call Recording**:
   - Use the file uploader to select an MP3 or M4A file from your local storage.

3. **Convert to WAV**:
   - After the audio upload, click the **Convert to WAV** button. The app will process the file and convert it to WAV format.

4. **Transcription**:
   - Once you have the WAV file, upload it to the app. Click the **Start Transcription** button to begin the transcription process. The app will display the transcription once it's complete.

5. **Analyze Transcription**:
   - After viewing the transcription, click the **Analyze Transcription** button to receive insights about the call, including evaluations on agent policies and effectiveness.

## 🎨 Example Screenshot

![Screenshot](https://via.placeholder.com/600x400.png?text=Screenshot) <!-- Replace with an actual screenshot of your app -->
## ⚙️ Technologies Used

- **Python**: The programming language used to build the application.
- **Streamlit**: A framework to create interactive web applications for machine learning and data science projects.
- **PyTorch**: A deep learning framework used to implement the Whisper and T5 models.
- **Transformers (Hugging Face)**: Pre-trained models for natural language processing tasks.
- **Torchaudio**: A library for audio processing in PyTorch.
- **FFmpeg**: A tool for handling multimedia files, enabling audio format conversion.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! To contribute to this project, please follow these steps:

1. **Fork the repository**: Click on the fork button at the top right of the page.
2. **Create your feature branch**: 
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**: 
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**: 
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**: Go to the original repository, click on "Pull Requests," and submit a new pull request.

## 📊 Future Improvements

- **Add More Models**: Integrate additional models for different languages or dialects.
- **User Authentication**: Implement user authentication for personalized experiences and data security.
- **Analytics Dashboard**: Create a dashboard for users to visualize call analytics and trends over time.
- **Mobile Compatibility**: Optimize the web application for mobile devices to enhance accessibility.

## 📫 Contact

For inquiries, please reach out:

- **Your Name**: [your.email@example.com](mailto:your.email@example.com)
- **GitHub**: [yourusername](https://github.com/yourusername)

## 🌟 Acknowledgements

- Thanks to the creators of the Whisper and T5 models for their contributions to the NLP community.
- Special thanks to the Streamlit team for providing a user-friendly interface for rapid application development.

```

### Customization Notes
- Replace placeholders like `yourusername`, `your.email@example.com`, and links to images/screenshots with your actual project details.
- Feel free to add or modify sections based on your project’s requirements, such as FAQs, tutorials, or a changelog.
