document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const imageInput = document.getElementById('image-upload');
    const fileNameSpan = document.getElementById('file-name');
    const uploadedImage = document.getElementById('uploaded-image');
    const imageCaption = document.getElementById('image-caption');
    const vsArena = document.getElementById('vs-arena');
    const cnnCard = document.getElementById('cnn-card');
    const vitCard = document.getElementById('vit-card');
    const cnnKO = document.getElementById('cnn-ko');
    const vitKO = document.getElementById('vit-ko');
    const cnnHealth = document.getElementById('cnn-health');
    const vitHealth = document.getElementById('vit-health');
    const vsBadge = document.getElementById('vs-badge');
    const resultsSection = document.getElementById('results-section'); // legacy, not used

    // Sound effect utility function
    function playSound(soundFile) {
        try {
            const audio = new Audio(`/static/${soundFile}`);
            audio.volume = 0.5; // Set volume to 50%
            audio.play().catch(e => {
                // Silently fail if sound file doesn't exist or can't play
                console.log(`Sound ${soundFile} could not be played:`, e.message);
            });
        } catch (e) {
            // Silently fail if sound file doesn't exist
            console.log(`Sound ${soundFile} not found:`, e.message);
        }
    }

    // Show selected file name
    imageInput.addEventListener('change', function() {
        fileNameSpan.textContent = imageInput.files[0] ? imageInput.files[0].name : '';
    });

    // Handle form submit
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        if (!imageInput.files[0]) return;
        const formData = new FormData();
        formData.append('image', imageInput.files[0]);
        // Animate out upload section
        uploadForm.parentElement.style.opacity = 1;
        uploadForm.parentElement.style.transition = 'opacity 0.5s';
        uploadForm.parentElement.style.opacity = 0;
        setTimeout(() => {
            uploadForm.parentElement.style.display = 'none';
            showVsArena();
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                    return;
                }
                renderResults(data);
            })
            .catch(() => alert('An error occurred during prediction.'));
        }, 500);
    });

    function showVsArena() {
        vsArena.style.display = 'grid';
        vsArena.style.opacity = 0;
        setTimeout(() => {
            vsArena.style.opacity = 1;
            // Play VS sound effect
            playSound('Vs.mp3');
        }, 100);
        // Animate VS badge
        vsBadge.classList.remove('hide');
    }

    function renderResults(data) {
        // Show uploaded image
        const imgUrl = URL.createObjectURL(imageInput.files[0]);
        uploadedImage.src = imgUrl;
        imageCaption.textContent = data.filename;

        // DR & DME labels
        document.getElementById('vit-dr-label').textContent = data.vit.dr.label;
        document.getElementById('vit-dme-label').textContent = data.vit.dme.label;
        document.getElementById('cnn-dr-label').textContent = data.cnn.dr.label;
        document.getElementById('cnn-dme-label').textContent = data.cnn.dme.label;

        // Gauges
        drawGauge('vit-dr-gauge', data.vit.dr.confidence, '#007bff');
        drawGauge('vit-dme-gauge', data.vit.dme.confidence, '#007bff');
        drawGauge('cnn-dr-gauge', data.cnn.dr.confidence, '#28a745');
        drawGauge('cnn-dme-gauge', data.cnn.dme.confidence, '#28a745');

        // Bar charts - FIXED: Correct number of classes for DR (5) and DME (3)
        // DR has 5 classes: No DR, Mild DR, Moderate DR, Severe DR, Proliferative DR
        drawBar('vit-dr-bar', data.vit.dr.distribution, ['No DR','Mild DR','Moderate DR','Severe DR','Proliferative DR'], '#007bff');
        drawBar('cnn-dr-bar', data.cnn.dr.distribution, ['No DR','Mild DR','Moderate DR','Severe DR','Proliferative DR'], '#28a745');

        // DME has 3 classes: No DME, Mild DME, Severe DME
        drawBar('vit-dme-bar', data.vit.dme.distribution, ['No DME','Mild DME','Severe DME'], '#007bff');
        drawBar('cnn-dme-bar', data.cnn.dme.distribution, ['No DME','Mild DME','Severe DME'], '#28a745');

        // Animate health bars
        animateHealthBar(cnnHealth, (data.cnn.dr.confidence + data.cnn.dme.confidence) / 2, 'cnn');
        animateHealthBar(vitHealth, (data.vit.dr.confidence + data.vit.dme.confidence) / 2, 'vit');

        // Determine winner
        setTimeout(() => {
            const cnnScore = (data.cnn.dr.confidence + data.cnn.dme.confidence) / 2;
            const vitScore = (data.vit.dr.confidence + data.vit.dme.confidence) / 2;
            if (cnnScore > vitScore) {
                showKO('cnn');
            } else if (vitScore > cnnScore) {
                showKO('vit');
            } else {
                // Draw: both glow
                cnnCard.classList.add('winner');
                vitCard.classList.add('winner');
            }
        }, 1800);
    }

    function showKO(winner) {
        // Play KO sound effect
        playSound('KO.mp3');

        if (winner === 'cnn') {
            cnnCard.classList.add('winner');
            vitCard.classList.add('loser');
            cnnKO.style.display = 'block';
            confettiBurst(cnnCard);
        } else {
            vitCard.classList.add('winner');
            cnnCard.classList.add('loser');
            vitKO.style.display = 'block';
            confettiBurst(vitCard);
        }
    }

    function animateHealthBar(bar, value, type) {
        bar.innerHTML = '';
        const inner = document.createElement('div');
        inner.className = 'health-bar-inner';
        inner.style.width = '0%';
        bar.appendChild(inner);
        setTimeout(() => {
            inner.style.width = Math.round(value * 100) + '%';
        }, 100);
    }

    function drawGauge(canvasId, value, color) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        ctx.clearRect(0, 0, 120, 120);
        // Animate gauge
        let start = 0;
        const end = value;
        const step = 0.02;
        function animate() {
            ctx.clearRect(0, 0, 120, 120);
            // Background circle
            ctx.beginPath();
            ctx.arc(60, 60, 50, 0, 2 * Math.PI);
            ctx.strokeStyle = '#e9ecef';
            ctx.lineWidth = 12;
            ctx.stroke();
            // Foreground arc
            ctx.beginPath();
            ctx.arc(60, 60, 50, -Math.PI/2, (2 * Math.PI * start) - Math.PI/2);
            ctx.strokeStyle = color;
            ctx.lineWidth = 12;
            ctx.stroke();
            // Text
            ctx.font = 'bold 1.2rem Inter, Roboto, Arial';
            ctx.fillStyle = color;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(Math.round(start*100) + '%', 60, 60);
            if (start < end) {
                start += step;
                if (start > end) start = end;
                requestAnimationFrame(animate);
            }
        }
        animate();
    }

    function drawBar(canvasId, data, labels, color) {
        const ctx = document.getElementById(canvasId).getContext('2d');

        // Destroy existing chart if it exists
        if (window[canvasId + '_chart']) {
            window[canvasId + '_chart'].destroy();
        }

        // Ensure data array matches labels array length
        const chartData = data.slice(0, labels.length);

        // Add debugging
        console.log(`Drawing chart for ${canvasId}:`, {
            labels: labels,
            data: chartData,
            labelsLength: labels.length,
            dataLength: chartData.length
        });

        window[canvasId + '_chart'] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    data: chartData.map(x => Math.round(x*100)),
                    backgroundColor: color + 'cc',
                    borderColor: color,
                    borderWidth: 2
                }]
            },
            options: {
                responsive: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { stepSize: 20 }
                    },
                    x: {
                        ticks: {
                            maxRotation: 45,
                            minRotation: 0
                        }
                    }
                }
            }
        });
    }

    // Confetti animation (simple burst)
    function confettiBurst(card) {
        const confetti = document.createElement('canvas');
        confetti.className = 'confetti';
        confetti.width = card.offsetWidth;
        confetti.height = card.offsetHeight;
        card.appendChild(confetti);
        const ctx = confetti.getContext('2d');
        const pieces = 32;
        const colors = ['#ffe066', '#ffb347', '#28a745', '#007bff', '#fff'];
        let particles = [];
        for (let i = 0; i < pieces; i++) {
            particles.push({
                x: confetti.width/2,
                y: confetti.height/2,
                r: Math.random()*8+4,
                c: colors[Math.floor(Math.random()*colors.length)],
                vx: (Math.random()-0.5)*8,
                vy: (Math.random()-0.5)*8,
                alpha: 1
            });
        }
        function draw() {
            ctx.clearRect(0,0,confetti.width,confetti.height);
            particles.forEach(p => {
                ctx.globalAlpha = p.alpha;
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r, 0, 2*Math.PI);
                ctx.fillStyle = p.c;
                ctx.fill();
            });
        }
        function update() {
            particles.forEach(p => {
                p.x += p.vx;
                p.y += p.vy;
                p.vy += 0.2;
                p.alpha -= 0.02;
            });
            particles = particles.filter(p => p.alpha > 0);
        }
        function loop() {
            draw();
            update();
            if (particles.length > 0) {
                requestAnimationFrame(loop);
            } else {
                confetti.remove();
            }
        }
        loop();
    }
});