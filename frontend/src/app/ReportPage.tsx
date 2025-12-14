import React, { useEffect, useState } from "react";
import SlantedButton from "./SlantedButton";
import { ChatMessage } from "./chatHistory";
import { useBackendServerUrl } from "./useBackendServerUrl";

interface EvaluationCategory {
    score: number;
    justification: string;
    missed_opportunity: string;
    feedback: string;
}

interface Recommendations {
    positive_aspects: string;
    areas_to_improve: string;
    patient_emotional_response: string;
}

interface ReportData {
    transcript: string;
    evaluation: {
        diagnosis_delivery: EvaluationCategory;
        emotional_acknowledgement: EvaluationCategory;
        language_accessibility: EvaluationCategory;
        emotional_responsiveness: EvaluationCategory;
        interaction_balance: EvaluationCategory;
        identity_future_preservation: EvaluationCategory;
    };
    recommendations: Recommendations;
}

interface ReportPageProps {
    chatHistory: ChatMessage[];
    onBack: () => void;
}

const ReportPage: React.FC<ReportPageProps> = ({ chatHistory, onBack }) => {
    const backendServerUrl = useBackendServerUrl();
    const [report, setReport] = useState<ReportData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const generateReport = async () => {
            try {
                // format transcript from chatHistory
                const transcript = chatHistory
                    .map((msg) => `${msg.role === "user" ? "Doctor" : "Patient"}: ${msg.content}`)
                    .join("\n");

                if (!backendServerUrl) throw new Error("Backend URL not found");

                const response = await fetch(`${backendServerUrl}/v1/report`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({ transcript }),
                });

                if (!response.ok) {
                    const errData = await response.json();
                    throw new Error(errData.detail || "Failed to generate report");
                }

                const data = await response.json();
                setReport(data);
            } catch (err: any) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        generateReport();
    }, [chatHistory, backendServerUrl]);

    const onDownloadHtml = async () => {
        try {
            if (!backendServerUrl || !report) return;
            const transcript = report.transcript;

            const response = await fetch(`${backendServerUrl}/v1/report/html`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ transcript }),
            });

            if (!response.ok) throw new Error("Failed to download HTML report");

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `unmute_report_${new Date().toISOString().slice(0, 19).replace(/:/g, "-")}.html`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (err: any) {
            setError(err.message);
        }
    };

    if (loading) {
        return (
            <div className="flex flex-col items-center justify-center min-h-screen text-white bg-background p-8">
                <h1 className="text-2xl animate-pulse">Generating Report...</h1>
                <p className="text-white/50 mt-4">Analyzing conversation with Mistral AI...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex flex-col items-center justify-center min-h-screen text-white bg-background p-8">
                <h1 className="text-2xl text-red-500">Error Generating Report</h1>
                <p className="text-white/70 mt-4">{error}</p>
                <SlantedButton onClick={onBack} kind="primary" extraClasses="mt-8">
                    Back to Start
                </SlantedButton>
            </div>
        );
    }

    if (!report) return null;

    const renderCategory = (title: string, category: EvaluationCategory) => (
        <div className="bg-white/5 p-6 rounded-lg mb-4 border border-white/10">
            <div className="flex justify-between items-center mb-2">
                <h3 className="text-xl font-bold text-white">{title}</h3>
                <div className={`px-3 py-1 rounded-full font-bold ${category.score >= 4 ? 'bg-green-500/20 text-green-400' : category.score >= 3 ? 'bg-yellow-500/20 text-yellow-400' : 'bg-red-500/20 text-red-400'}`}>
                    Score: {category.score}/5
                </div>
            </div>
            <p className="text-white/80 mb-2">{category.justification}</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4 text-sm">
                <div>
                    <span className="text-red-400 font-semibold block mb-1">Missed Opportunity:</span>
                    <span className="text-white/60">{category.missed_opportunity}</span>
                </div>
                <div>
                    <span className="text-blue-400 font-semibold block mb-1">Feedback:</span>
                    <span className="text-white/60">{category.feedback}</span>
                </div>
            </div>
        </div>
    );

    return (
        <div className="min-h-screen bg-background text-offwhite p-4 md:p-8 overflow-y-auto w-full">
            <div className="max-w-4xl mx-auto">
                <div className="flex justify-between items-center mb-8">
                    <div>
                        <h1 className="text-3xl font-bold text-gradium-orange mb-2">Detailed Evaluation Report</h1>
                        <button
                            onClick={onDownloadHtml}
                            className="text-sm text-white/50 hover:text-white underline cursor-pointer"
                        >
                            Download HTML Report
                        </button>
                    </div>
                    <SlantedButton onClick={onBack} kind="secondary">
                        Start New Session
                    </SlantedButton>
                </div>

                <div className="space-y-8">
                    {/* Recommendations Section */}
                    <section className="bg-white/10 p-6 rounded-xl border border-white/20">
                        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                            📝 Key Recommendations
                        </h2>
                        <div className="grid grid-cols-1 gap-4">
                            <div className="p-4 bg-green-900/20 rounded-lg border border-green-500/30">
                                <h4 className="font-bold text-green-400 mb-2">Positive Aspects</h4>
                                <p>{report.recommendations.positive_aspects}</p>
                            </div>
                            <div className="p-4 bg-red-900/20 rounded-lg border border-red-500/30">
                                <h4 className="font-bold text-red-400 mb-2">Areas to Improve</h4>
                                <p>{report.recommendations.areas_to_improve}</p>
                            </div>
                            <div className="p-4 bg-blue-900/20 rounded-lg border border-blue-500/30">
                                <h4 className="font-bold text-blue-400 mb-2">Patient's Emotional Response</h4>
                                <p>{report.recommendations.patient_emotional_response}</p>
                            </div>
                        </div>
                    </section>

                    {/* Detailed Scores */}
                    <section>
                        <h2 className="text-2xl font-bold mb-4">Detailed Analysis</h2>
                        {renderCategory("Diagnosis Delivery", report.evaluation.diagnosis_delivery)}
                        {renderCategory("Emotional Acknowledgement", report.evaluation.emotional_acknowledgement)}
                        {renderCategory("Language Accessibility", report.evaluation.language_accessibility)}
                        {renderCategory("Emotional Responsiveness", report.evaluation.emotional_responsiveness)}
                        {renderCategory("Interaction Balance", report.evaluation.interaction_balance)}
                        {renderCategory("Identity & Future Preservation", report.evaluation.identity_future_preservation)}
                    </section>

                    {/* Transcript */}
                    <section className="opacity-70 mt-12">
                        <h3 className="text-lg font-bold mb-2">Transcript Reference</h3>
                        <pre className="whitespace-pre-wrap bg-black/30 p-4 rounded text-xs font-mono border border-white/10">
                            {report.transcript}
                        </pre>
                    </section>
                </div>
            </div>
        </div>
    );
};

export default ReportPage;
