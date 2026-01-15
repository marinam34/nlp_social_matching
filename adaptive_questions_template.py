"""
Simple HTML template for adaptive questions page
Questions are generated dynamically based on previous answers
"""

ADAPTIVE_QUESTIONS_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Connect - Tell Us About Yourself</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .container {
            width: 100%;
            max-width: 700px;
        }
        .progress-card {
            background: white;
            border-radius: 20px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .progress-bar {
            height: 10px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s;
            width: 0%;
        }
        .progress-text {
            margin-top: 10px;
            text-align: center;
            color: #666;
            font-size: 14px;
            font-weight: 500;
        }
        .question-card {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            animation: slideIn 0.5s;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .question-header {
            font-size: 14px;
            color: #667eea;
            font-weight: 600;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .question-text {
            font-size: 22px;
            color: #333;
            line-height: 1.5;
            margin-bottom: 30px;
        }
        textarea {
            width: 100%;
            min-height: 150px;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            font-size: 16px;
            font-family: 'Poppins', sans-serif;
            resize: vertical;
            transition: all 0.3s;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        .char-count {
            text-align: right;
            color: #999;
            font-size: 13px;
            margin-top: 8px;
        }
        .buttons {
            display: flex;
            gap: 15px;
            margin-top: 30px;
        }
        .btn {
            flex: 1;
            padding: 16px;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            font-family: 'Poppins', sans-serif;
        }
        .btn-back {
            background: #f0f0f0;
            color: #666;
        }
        .btn-next {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        .loading {
            text-align: center;
            color: white;
            padding: 40px;
        }
        .spinner {
            border: 4px solid rgba(255,255,255,0.3);
            border-top: 4px solid white;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="progress-card">
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <div class="progress-text" id="progressText">Question 1 of 5</div>
        </div>

        <div id="loadingDiv" class="loading" style="display: none;">
            <div class="spinner"></div>
            <p>Getting your next question...</p>
        </div>

        <div id="questionCard" class="question-card">
            <div class="question-header" id="questionHeader">QUESTION 1</div>
            <div class="question-text" id="questionText"></div>
            <textarea id="answerInput" placeholder="Share your thoughts here..." maxlength="1000"></textarea>
            <div class="char-count"><span id="charCount">0</span>/1000</div>
            <div class="buttons">
                <button class="btn btn-back" id="backBtn" onclick="goBack()" disabled>Back</button>
                <button class="btn btn-next" id="nextBtn" onclick="goNext()" disabled>Next</button>
            </div>
        </div>
    </div>

    <script>
        const userId = new URLSearchParams(window.location.search).get('user_id');
        let currentQuestion = null;
        let allAnswers = [];
        let maxQuestions = 7;  // 3 fixed + 4 generated

        // Character count
        document.getElementById('answerInput').addEventListener('input', function() {
            const count = this.value.length;
            document.getElementById('charCount').textContent = count;
            document.getElementById('nextBtn').disabled = false; // Always enabled
        });

        async function loadFirstQuestion() {
            try {
                const response = await fetch('/api/get-first-question', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({user_id: userId})
                });

                const data = await response.json();
                if (data.question) {
                    displayQuestion(data.question);
                }
            } catch (error) {
                console.error('Error loading question:', error);
                alert('Error loading questions. Please try again.');
            }
        }

        function displayQuestion(question) {
            currentQuestion = question;
            document.getElementById('questionText').textContent = question.question;
            document.getElementById('answerInput').value = '';
            document.getElementById('charCount').textContent = '0';
            document.getElementById('nextBtn').disabled = true;
            
            updateProgress();
            document.getElementById('answerInput').focus();
        }

        function updateProgress() {
            const questionNum = allAnswers.length + 1;
            const progress = (questionNum / maxQuestions) * 100;
            
            document.getElementById('progressFill').style.width = progress + '%';
            document.getElementById('progressText').textContent = `Question ${questionNum} of ${maxQuestions}`;
            document.getElementById('questionHeader').textContent = `QUESTION ${questionNum}`;
            document.getElementById('backBtn').disabled = allAnswers.length === 0;
        }

        async function goNext() {
            const answer = document.getElementById('answerInput').value.trim();
            
            if (answer.length === 0) {
                alert('Please provide an answer');
                return;
            }

            // Save current answer
            allAnswers.push({
                question: currentQuestion.question,
                answer: answer
            });

            // Check if we've reached max questions
            if (allAnswers.length >= maxQuestions) {
                await finishQuestions();
                return;
            }

            // Get next question
            document.getElementById('questionCard').style.display = 'none';
            document.getElementById('loadingDiv').style.display = 'block';

            try {
                const response = await fetch('/api/get-next-question', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        user_id: userId,
                        previous_answers: allAnswers
                    })
                });

                const data = await response.json();
                
                document.getElementById('loadingDiv').style.display = 'none';
                document.getElementById('questionCard').style.display = 'block';

                if (data.question) {
                    displayQuestion(data.question);
                } else {
                    // No more questions
                    await finishQuestions();
                }
            } catch (error) {
                console.error('Error getting next question:', error);
                alert('Error loading next question');
                document.getElementById('loadingDiv').style.display = 'none';
                document.getElementById('questionCard').style.display = 'block';
            }
        }

        function goBack() {
            if (allAnswers.length > 0) {
                allAnswers.pop();
                updateProgress();
                // You would need to store questions to go back properly
                // For simplicity, we'll just clear the current answer
                document.getElementById('answerInput').value = '';
                document.getElementById('charCount').textContent = '0';
                document.getElementById('nextBtn').disabled = true;
            }
        }

        async function finishQuestions() {
            document.getElementById('questionCard').style.display = 'none';
            document.getElementById('loadingDiv').style.display = 'block';
            document.querySelector('.loading p').textContent = 'Analyzing your profile...';

            try {
                const response = await fetch('/api/complete-questions', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        user_id: userId,
                        answers: allAnswers
                    })
                });

                if (response.ok) {
                    // Redirect to matches page
                    window.location.href = '/matches?user_id=' + userId;
                } else {
                    alert('Error processing answers');
                }
            } catch (error) {
                console.error('Error completing questions:', error);
                alert('Error processing answers');
            }
        }

        // Load first question on page load
        loadFirstQuestion();
    </script>
</body>
</html>
"""
