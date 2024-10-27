document.getElementById('sentimentForm').addEventListener('submit', function (e) {
    e.preventDefault();  // Prevent form from reloading the page
    
    // Get the input text from the form
    const text = document.getElementById('text').value;
    
    // Send a POST request to the Flask backend
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        // Display the sentiment result along with character and word counts, and language detection
        document.getElementById('result').innerHTML = 
            `Sentiment: ${data.sentiment}<br>
            Character Count: ${data.char_count}<br>
            Word Count: ${data.word_count}<br>
            Language: ${data.language}`;
    })
    .catch(error => {
        console.error('Error:', error);
    });    
});