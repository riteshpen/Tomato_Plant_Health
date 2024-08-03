document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const predictButton = document.getElementById('predict-button');
    const resultDiv = document.getElementById('result');
    const predictionSpan = document.getElementById('prediction');
    const confidenceSpan = document.getElementById('confidence');
    const errorMessageDiv = document.getElementById('error-message');

    predictButton.addEventListener('click', async function() {
        if (!fileInput.files.length) {
            alert('Please select an image file first.');
            return;
        }

        const formData = new FormData(form);
        
        try {
            const response = await fetch('/', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if (data.error) {
                console.error('Error:', data.error);
                errorMessageDiv.textContent = 'Error: ' + data.error;
                errorMessageDiv.classList.remove('hidden');
                resultDiv.classList.add('hidden');
            } else {
                predictionSpan.textContent = data.prediction;
                confidenceSpan.textContent = (data.confidence * 100).toFixed(2);
                resultDiv.classList.remove('hidden');
                errorMessageDiv.classList.add('hidden');
            }
        } catch (error) {
            console.error('Error:', error);
            errorMessageDiv.textContent = 'An error occurred while processing the image.';
            errorMessageDiv.classList.remove('hidden');
            resultDiv.classList.add('hidden');
        }
    });
});