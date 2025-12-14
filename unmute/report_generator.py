
def generate_html(report_data):
    """
    Generates a static HTML string from the report JSON data.
    """
    
    # Extract data for easier access
    evaluation = report_data.get("evaluation", {})
    recommendations = report_data.get("recommendations", {})
    transcript = report_data.get("transcript", "")
    
    # Helper to determine color based on score
    def get_score_color(score):
        if score >= 4:
            return "#4ade80" # green-400
        elif score >= 3:
            return "#facc15" # yellow-400
        else:
            return "#f87171" # red-400

    # Build HTML sections for evaluation categories
    evaluation_html = ""
    for category_key, category_data in evaluation.items():
        title = category_key.replace("_", " ").title()
        score = category_data.get("score", 0)
        color = get_score_color(score)
        
        evaluation_html += f"""
        <div class="category">
            <div class="category-header">
                <h3>{title}</h3>
                <span class="score" style="color: {color}; background-color: {color}20;">Score: {score}/5</span>
            </div>
            <p class="justification">{category_data.get("justification", "")}</p>
            <div class="feedback-grid">
                <div class="feedback-box">
                    <span class="label text-red">Missed Opportunity:</span>
                    <span class="text">{category_data.get("missed_opportunity", "")}</span>
                </div>
                <div class="feedback-box">
                    <span class="label text-blue">Feedback:</span>
                    <span class="text">{category_data.get("feedback", "")}</span>
                </div>
            </div>
        </div>
        """

    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unmute Evaluation Report</title>
    <style>
        :root {{
            --bg-color: #1a1a1a;
            --text-color: #e5e5e5;
            --card-bg: #262626;
            --border-color: #404040;
            --accent-orange: #f97316;
        }}
        body {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            margin: 0;
            padding: 2rem;
        }}
        .container {{
            max_width: 900px;
            margin: 0 auto;
        }}
        h1 {{
            color: var(--accent-orange);
            font-size: 2.5rem;
            margin-bottom: 2rem;
            text-align: center;
        }}
        h2 {{
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 0.5rem;
            margin-top: 3rem;
            margin-bottom: 1.5rem;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 1.5rem;
            margin-bottom: 3rem;
        }}
        @media (min-width: 768px) {{
            .summary-grid {{
                grid-template-columns: repeat(3, 1fr);
            }}
        }}
        .summary-card {{
            background: var(--card-bg);
            padding: 1.5rem;
            border-radius: 0.75rem;
            border: 1px solid var(--border-color);
        }}
        .summary-card h4 {{
            margin-top: 0;
            margin-bottom: 1rem;
            font-size: 1.1rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .summary-card.positive h4 {{ color: #4ade80; }}
        .summary-card.improve h4 {{ color: #f87171; }}
        .summary-card.emotional h4 {{ color: #60a5fa; }}
        
        .category {{
            background: var(--card-bg);
            padding: 1.5rem;
            border-radius: 0.75rem;
            margin-bottom: 1.5rem;
            border: 1px solid var(--border-color);
        }}
        .category-header {{
            display: flex;
            justify_content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }}
        .category-header h3 {{
            margin: 0;
            font-size: 1.25rem;
        }}
        .score {{
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-weight: bold;
        }}
        .justification {{
            color: #a3a3a3;
            margin-bottom: 1.5rem;
        }}
        .feedback-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 1rem;
        }}
        @media (min-width: 768px) {{
            .feedback-grid {{
                grid-template-columns: 1fr 1fr;
            }}
        }}
        .feedback-box {{
            font-size: 0.9rem;
        }}
        .label {{
            display: block;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }}
        .text-red {{ color: #f87171; }}
        .text-blue {{ color: #60a5fa; }}
        
        .transcript-box {{
            background-color: #000000;
            padding: 1.5rem;
            border-radius: 0.5rem;
            font-family: monospace;
            white-space: pre-wrap;
            font-size: 0.9rem;
            color: #d4d4d4;
            max-height: 500px;
            overflow-y: auto;
            border: 1px solid var(--border-color);
        }}
        .footer {{
            margin-top: 4rem;
            text-align: center;
            color: #737373;
            font-size: 0.875rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Detailed Evaluation Report</h1>

        <!-- Recommendations Summary -->
        <div class="summary-grid">
            <div class="summary-card positive">
                <h4>Positive Aspects</h4>
                <p>{recommendations.get("positive_aspects", "")}</p>
            </div>
            <div class="summary-card improve">
                <h4>Areas to Improve</h4>
                <p>{recommendations.get("areas_to_improve", "")}</p>
            </div>
            <div class="summary-card emotional">
                <h4>Emotional Response</h4>
                <p>{recommendations.get("patient_emotional_response", "")}</p>
            </div>
        </div>

        <!-- Detailed Analysis -->
        <h2>Detailed Analysis</h2>
        {evaluation_html}

        <!-- Transcript -->
        <h2>Transcript Reference</h2>
        <div class="transcript-box">
{transcript}
        </div>

        <div class="footer">
            Generated by Unmute
        </div>
    </div>
</body>
</html>
    """
    return html_template
