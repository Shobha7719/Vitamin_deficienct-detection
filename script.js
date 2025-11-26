document.addEventListener('DOMContentLoaded', () => {
    // Flash message dismissal
    const flashMessages = document.querySelectorAll('.alert');
    flashMessages.forEach(msg => {
        setTimeout(() => {
            msg.style.opacity = '0';
            msg.style.transform = 'translateY(-20px)';
            setTimeout(() => msg.remove(), 500);
        }, 5000); // Messages disappear after 5 seconds
    });

    // Handle file input display
    const imageUpload = document.getElementById('imageUpload');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    if (imageUpload && fileNameDisplay) {
        imageUpload.addEventListener('change', function() {
            if (this.files && this.files.length > 0) {
                fileNameDisplay.textContent = this.files[0].name;
            } else {
                fileNameDisplay.textContent = 'No file chosen';
            }
        });
    }

    // Camera functionality
    const startCameraButton = document.getElementById('startCameraButton');
    const videoFeed = document.getElementById('videoFeed');
    const captureButton = document.getElementById('captureButton');
    const canvas = document.getElementById('canvas');
    const cameraStatus = document.getElementById('cameraStatus');
    const cameraFeedContainer = document.querySelector('.camera-feed-container');
    const cameraPredictionResults = document.getElementById('cameraPredictionResults');
    const cameraPredictionImage = document.getElementById('cameraPredictionImage');
    const cameraPredictedClass = document.getElementById('cameraPredictedClass');
    const cameraConfidenceScore = document.getElementById('cameraConfidenceScore');
    const cameraConfidenceBar = document.getElementById('cameraConfidenceBar');
    const cameraSpinner = document.getElementById('cameraSpinner');
    const camKnowMoreLinkContainer = document.getElementById('camKnowMoreLinkContainer'); // Get the container

    let stream = null;

    if (startCameraButton) {
        startCameraButton.addEventListener('click', async () => {
            if (stream) {
                // If camera is already running, stop it
                stream.getTracks().forEach(track => track.stop());
                videoFeed.srcObject = null;
                cameraFeedContainer.style.display = 'none';
                captureButton.style.display = 'none';
                startCameraButton.textContent = 'Start Camera';
                cameraStatus.textContent = '';
                cameraPredictionResults.style.display = 'none';
                camKnowMoreLinkContainer.innerHTML = ''; // Clear previous link
                camKnowMoreLinkContainer.style.display = 'none'; // Hide link container
                stream = null;
                return;
            }

            cameraStatus.textContent = 'Requesting camera access...';
            cameraSpinner.style.display = 'block';
            // Hide any previous camera prediction results and know more link
            cameraPredictionResults.style.display = 'none';
            camKnowMoreLinkContainer.innerHTML = '';
            camKnowMoreLinkContainer.style.display = 'none';

            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: true });
                videoFeed.srcObject = stream;
                cameraFeedContainer.style.display = 'block';
                captureButton.style.display = 'block';
                startCameraButton.textContent = 'Stop Camera';
                cameraStatus.textContent = 'Camera is active.';
                cameraSpinner.style.display = 'none';
                // Hide existing prediction results if any from file upload
                const uploadedPredictionContainer = document.querySelector('.prediction-results-container:not(#cameraPredictionResults)');
                if (uploadedPredictionContainer) {
                    uploadedPredictionContainer.style.display = 'none';
                }


            } catch (err) {
                console.error('Error accessing camera:', err);
                cameraStatus.textContent = 'Could not access camera. Please ensure permissions are granted.';
                cameraFeedContainer.style.display = 'none';
                captureButton.style.display = 'none';
                cameraSpinner.style.display = 'none';
            }
        });
    }

    if (captureButton) {
        captureButton.addEventListener('click', () => {
            if (!stream) {
                cameraStatus.textContent = 'Camera not active.';
                return;
            }

            const context = canvas.getContext('2d');
            canvas.width = videoFeed.videoWidth;
            canvas.height = videoFeed.videoHeight;
            context.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);

            cameraStatus.textContent = 'Capturing image and predicting...';
            cameraSpinner.style.display = 'block';

            // Clear previous "Know More" link content and hide it
            camKnowMoreLinkContainer.innerHTML = '';
            camKnowMoreLinkContainer.style.display = 'none';

            // Convert canvas content to Blob and send to server
            canvas.toBlob(async (blob) => {
                const formData = new FormData();
                formData.append('image', blob, 'captured_image.png');

                try {
                    const response = await fetch('/predict_camera', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();

                    if (response.ok) {
                        cameraPredictedClass.textContent = result.predicted_class;
                        cameraConfidenceScore.textContent = result.confidence;
                        cameraPredictionImage.src = result.image_url;
                        const confidenceValue = parseFloat(result.confidence.replace('%', ''));
                        cameraConfidenceBar.style.width = `${confidenceValue}%`;
                        cameraPredictionResults.style.display = 'block';
                        cameraStatus.textContent = 'Prediction complete!';

                        // --- ADDED/MODIFIED: "Know More" link logic for camera prediction ---
                        if (result.deficiency_details) { // Assuming your Flask endpoint returns 'deficiency_details'
                            const knowMoreLink = document.createElement('a');
                            knowMoreLink.href = `/deficiency/${result.deficiency_details}`; // Construct the URL
                            knowMoreLink.classList.add('know-more-btn');
                            knowMoreLink.innerHTML = `<i class="fas fa-info-circle"></i> Know More About ${result.predicted_class} Deficiency`;
                            camKnowMoreLinkContainer.appendChild(knowMoreLink);
                            camKnowMoreLinkContainer.style.display = 'block'; // Make the container visible
                        } else {
                            const genericMessage = document.createElement('p');
                            genericMessage.style.marginTop = '20px';
                            genericMessage.style.color = 'var(--light-text-color)';
                            genericMessage.textContent = 'No specific information link available for this prediction.';
                            camKnowMoreLinkContainer.appendChild(genericMessage);
                            camKnowMoreLinkContainer.style.display = 'block'; // Make the container visible for the message
                        }
                        // --- END ADDED/MODIFIED ---

                    } else {
                        cameraStatus.textContent = `Prediction failed: ${result.error || 'Unknown error'}`;
                        cameraPredictionResults.style.display = 'none'; // Hide results on error
                        camKnowMoreLinkContainer.innerHTML = ''; // Clear link on error
                        camKnowMoreLinkContainer.style.display = 'none'; // Hide link container on error
                    }
                } catch (error) {
                    console.error('Error during camera prediction fetch:', error);
                    cameraStatus.textContent = `Error sending image: ${error.message}`;
                    cameraPredictionResults.style.display = 'none'; // Hide results on error
                    camKnowMoreLinkContainer.innerHTML = ''; // Clear link on error
                    camKnowMoreLinkContainer.style.display = 'none'; // Hide link container on error
                } finally {
                    cameraSpinner.style.display = 'none';
                }
            }, 'image/png');
        });
    }

    // Initial check for existing prediction results from server-side render (for file upload)
    const initialPredictionContainer = document.querySelector('.prediction-results-container:not(#cameraPredictionResults)');
    if (initialPredictionContainer && initialPredictionContainer.querySelector('.predicted-class').textContent.trim() !== '') {
        const confidenceElement = initialPredictionContainer.querySelector('.confidence-score');
        const confidenceText = confidenceElement.textContent;
        const confidenceValue = parseFloat(confidenceText.replace('%', ''));
        const confidenceBar = initialPredictionContainer.querySelector('.confidence-bar');
        if (confidenceBar) {
            confidenceBar.style.width = `${confidenceValue}%`;
        }
    }
});