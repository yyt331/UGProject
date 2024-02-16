document.getElementById('uploadForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const formData = new FormData(this);
    fetch('/analyze', { method: 'POST', body: formData })
        .then(response => response.json())
        .then(data => {
            document.getElementById('result').textContent = `Result: ${data.result}, Confidence: ${data.confidence}%`;
        });
});