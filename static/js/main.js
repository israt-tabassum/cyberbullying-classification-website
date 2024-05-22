document.addEventListener('DOMContentLoaded', function() {
    var form = document.getElementById('fileUploadForm');
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();

            const formData = new FormData(this); // 'this' refers to the form

            // Append additional data if needed
            // const additionalData = document.getElementById('someElement').value;
            // formData.append('additionalData', additionalData);

            fetch('/predict', {
                method: 'POST',
                body: formData,
            })
            .then(response => response.json())
            .then(data => {
                console.log(data);
                // Update the results in the HTML
                document.getElementById('textResult').innerHTML = 'Text label: ' + data.text_label;
                document.getElementById('imageResult').innerHTML = 'Image label: ' + data.image_label;
            })
            .catch(error => {
                console.error('Error:', error);
            });
        });
    } else {
        console.error('Form not found');
    }
});
