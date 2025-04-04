document.addEventListener('DOMContentLoaded', function() {
    // Toggle advanced options
    const toggleButton = document.getElementById('toggle-advanced');
    const advancedFields = document.querySelectorAll('.advanced-fields');
    
    if (toggleButton) {
        toggleButton.addEventListener('click', function() {
            advancedFields.forEach(field => {
                if (field.style.display === 'none') {
                    field.style.display = 'block';
                    toggleButton.textContent = 'Hide Advanced Options';
                } else {
                    field.style.display = 'none';
                    toggleButton.textContent = 'Show Advanced Options';
                }
            });
        });
    }
    
    // Form validation
    const predictionForm = document.getElementById('prediction-form');
    
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(event) {
            const age = document.getElementById('age');
            
            if (age && (age.value < 18 || age.value > 100)) {
                event.preventDefault();
                alert('Please enter a valid age between 18 and 100.');
                age.focus();
            }
        });
    }
    
    // Highlight active navigation link
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('nav a');
    
    navLinks.forEach(link => {
        const linkPath = link.getAttribute('href');
        if (currentPath === linkPath) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
});