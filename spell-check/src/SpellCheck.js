import React, { useState, useRef } from "react";
import axios from "axios";

const SpellCheck = () => {
    const [referenceText, setReferenceText] = useState("");
    const [language, setLanguage] = useState("english");
    const [recording, setRecording] = useState(false);
    const [audioBlob, setAudioBlob] = useState(null);
    const [transcribedText, setTranscribedText] = useState("");
    const [accuracy, setAccuracy] = useState(null);
    const [incorrectWords, setIncorrectWords] = useState([]);
    
    const mediaRecorderRef = useRef(null);
    const chunksRef = useRef([]);

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream);

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    chunksRef.current.push(event.data);
                }
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(chunksRef.current, { type: "audio/webm" });
                setAudioBlob(audioBlob);
                chunksRef.current = [];
            };

            mediaRecorderRef.current = mediaRecorder;
            mediaRecorder.start();
            setRecording(true);
        } catch (error) {
            console.error("Lỗi khi ghi âm:", error);
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current) {
            mediaRecorderRef.current.stop();
            setRecording(false);
        }
    };

    const uploadAudio = async () => {
        if (!audioBlob) {
            alert("Không có file ghi âm nào để gửi!");
            return;
        }

        if (!referenceText.trim()) {
            alert("Vui lòng nhập câu tham chiếu!");
            return;
        }

        const formData = new FormData();
        formData.append("file", audioBlob, "audio.webm");
        formData.append("reference_text", referenceText);
        formData.append("language", language);

        try {
            const response = await axios.post("http://localhost:5000/upload", formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });

            setTranscribedText(response.data.transcribed_text);
            setAccuracy(response.data.accuracy);
            setIncorrectWords(response.data.incorrect_words);
        } catch (error) {
            console.error("Lỗi khi gửi file:", error);
        }
    };

    return (
        <div style={{ textAlign: "center", padding: "20px" }}>
            <h2>SpellCheck</h2>

            <br /><br />

            <label>
                <strong>Nhập câu cần đọc:</strong>
            </label>
            <input
                type="text"
                value={referenceText}
                onChange={(e) => setReferenceText(e.target.value)}
                placeholder="Nhập câu..."
                style={{ margin: "10px", padding: "5px", width: "80%" }}
            />

            <br />

            <button onClick={recording ? stopRecording : startRecording} style={{ margin: "10px", padding: "10px" }}>
                {recording ? "Dừng ghi âm" : "Bắt đầu ghi âm"}
            </button>

            <button onClick={uploadAudio} disabled={!audioBlob} style={{ margin: "10px", padding: "10px" }}>
                Gửi lên server
            </button>

            {audioBlob && (
                <div>
                    <h3>🎧 Nghe lại ghi âm:</h3>
                    <audio controls src={URL.createObjectURL(audioBlob)}></audio>
                </div>
            )}

            {transcribedText && (
                <div>
                    <h3>Kết quả:</h3>
                    <p><strong>Văn bản nhận diện:</strong> {transcribedText}</p>
                    <p><strong>Độ chính xác:</strong> {accuracy !== null ? (accuracy * 100).toFixed(2) + "%" : "N/A"}</p>
                    {incorrectWords.length > 0 && (
                        <p><strong>Từ sai:</strong> {incorrectWords.join(", ")}</p>
                    )}
                </div>
            )}
        </div>
    );
};

export default SpellCheck;
