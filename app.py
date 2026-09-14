import pickle
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Risk Assessment Portal")

# Load model
with open("logistic.pkl", "rb") as f:
    model = pickle.load(f)

class StudentData(BaseModel):
    attendance: float = Field(..., ge=0, le=100)
    study_hours: float = Field(..., ge=0)
    past_failures: int = Field(..., ge=0)
    assignments_completed_pct: float = Field(..., ge=0, le=100)
    parental_education: int = Field(..., ge=0, le=5)  # Categorical encoded
    family_income: int = Field(..., ge=0, le=5)       # Categorical encoded
    extracurricular: int = Field(..., ge=0, le=1)     # 0 or 1
    internet_access: int = Field(..., ge=0, le=1)    # 0 or 1
    previous_grade: float = Field(..., ge=0, le=100)
    final_score: float = Field(..., ge=0, le=100)

@app.post("/predict")
def predict(data: StudentData):
    features = np.array([[
        data.attendance,
        data.study_hours,
        data.past_failures,
        data.assignments_completed_pct,
        data.parental_education,
        data.family_income,
        data.extracurricular,
        data.internet_access,
        data.previous_grade,
        data.final_score
    ]])
    
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    classes = model.classes_.tolist()
    
    prob_dict = {classes[i]: round(float(probabilities[i]) * 100, 2) for i in range(len(classes))}
    
    return {
        "status": "success",
        "prediction": str(prediction),
        "probabilities": prob_dict
    }

@app.get("/", response_class=HTMLResponse)
def serve_react_app():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Student Risk Classifier</title>
        <script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
        <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
        <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Plus Jakarta Sans', sans-serif; }
        </style>
    </head>
    <body class="bg-slate-950 text-slate-100 min-h-screen">
        <div id="root"></div>

        <script type="text/babel">
            const { useState } = React;

            function App() {
                const [formData, setFormData] = useState({
                    attendance: 85,
                    study_hours: 15,
                    past_failures: 0,
                    assignments_completed_pct: 90,
                    parental_education: 3,
                    family_income: 3,
                    extracurricular: 1,
                    internet_access: 1,
                    previous_grade: 78,
                    final_score: 82
                });

                const [result, setResult] = useState(null);
                const [loading, setLoading] = useState(false);

                const handleChange = (e) => {
                    const { name, value } = e.target;
                    setFormData(prev => ({ ...prev, [name]: parseFloat(value) || 0 }));
                };

                const handleSubmit = async (e) => {
                    e.preventDefault();
                    setLoading(true);
                    try {
                        const response = await fetch('/predict', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(formData)
                        });
                        const data = await response.json();
                        setResult(data);
                    } catch (err) {
                        alert("Prediction request failed.");
                    } finally {
                        setLoading(false);
                    }
                };

                const getBadgeColor = (status) => {
                    if (status === 'Safe') return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
                    if (status === 'At-Risk') return 'bg-amber-500/10 text-amber-400 border-amber-500/30';
                    return 'bg-rose-500/10 text-rose-400 border-rose-500/30';
                };

                return (
                    <div className="max-w-6xl mx-auto px-6 py-12">
                        <header className="mb-10 text-center">
                            <h1 className="text-4xl font-bold tracking-tight text-white mb-2">Student Performance Analytics</h1>
                            <p className="text-slate-400">ML-driven risk evaluation and performance prediction platform</p>
                        </header>

                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                            <form onSubmit={handleSubmit} className="lg:col-span-2 bg-slate-900/50 border border-slate-800 rounded-2xl p-6 backdrop-blur-sm shadow-xl">
                                <h2 className="text-xl font-semibold mb-6 text-slate-200">Input Metrics</h2>
                                
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                                    <div>
                                        <label className="block text-xs font-medium text-slate-400 mb-1">Attendance (%)</label>
                                        <input type="number" name="attendance" value={formData.attendance} onChange={handleChange} className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-indigo-500" />
                                    </div>
                                    <div>
                                        <label className="block text-xs font-medium text-slate-400 mb-1">Weekly Study Hours</label>
                                        <input type="number" name="study_hours" value={formData.study_hours} onChange={handleChange} className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-indigo-500" />
                                    </div>
                                    <div>
                                        <label className="block text-xs font-medium text-slate-400 mb-1">Past Failures</label>
                                        <input type="number" name="past_failures" value={formData.past_failures} onChange={handleChange} className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-indigo-500" />
                                    </div>
                                    <div>
                                        <label className="block text-xs font-medium text-slate-400 mb-1">Assignments Completed (%)</label>
                                        <input type="number" name="assignments_completed_pct" value={formData.assignments_completed_pct} onChange={handleChange} className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-indigo-500" />
                                    </div>
                                    <div>
                                        <label className="block text-xs font-medium text-slate-400 mb-1">Parental Education Level (0-5)</label>
                                        <input type="number" name="parental_education" value={formData.parental_education} onChange={handleChange} className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-indigo-500" />
                                    </div>
                                    <div>
                                        <label className="block text-xs font-medium text-slate-400 mb-1">Family Income Level (0-5)</label>
                                        <input type="number" name="family_income" value={formData.family_income} onChange={handleChange} className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-indigo-500" />
                                    </div>
                                    <div>
                                        <label className="block text-xs font-medium text-slate-400 mb-1">Extracurricular Activity</label>
                                        <select name="extracurricular" value={formData.extracurricular} onChange={handleChange} className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-indigo-500">
                                            <option value={1}>Yes</option>
                                            <option value={0}>No</option>
                                        </select>
                                    </div>
                                    <div>
                                        <label className="block text-xs font-medium text-slate-400 mb-1">Internet Access</label>
                                        <select name="internet_access" value={formData.internet_access} onChange={handleChange} className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-indigo-500">
                                            <option value={1}>Yes</option>
                                            <option value={0}>No</option>
                                        </select>
                                    </div>
                                    <div>
                                        <label className="block text-xs font-medium text-slate-400 mb-1">Previous Grade</label>
                                        <input type="number" name="previous_grade" value={formData.previous_grade} onChange={handleChange} className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-indigo-500" />
                                    </div>
                                    <div>
                                        <label className="block text-xs font-medium text-slate-400 mb-1">Current Score</label>
                                        <input type="number" name="final_score" value={formData.final_score} onChange={handleChange} className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-indigo-500" />
                                    </div>
                                </div>

                                <button type="submit" disabled={loading} className="mt-6 w-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium py-2.5 rounded-lg transition duration-200">
                                    {loading ? 'Evaluating...' : 'Run Prediction'}
                                </button>
                            </form>

                            <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between">
                                <div>
                                    <h2 className="text-xl font-semibold mb-6 text-slate-200">Assessment Result</h2>
                                    {result ? (
                                        <div className="space-y-6">
                                            <div>
                                                <span className="text-xs text-slate-400 uppercase tracking-wider block mb-2">Class Output</span>
                                                <div className={`inline-block px-4 py-2 rounded-xl text-lg font-semibold border ${getBadgeColor(result.prediction)}`}>
                                                    {result.prediction}
                                                </div>
                                            </div>
                                            <div>
                                                <span className="text-xs text-slate-400 uppercase tracking-wider block mb-3">Probability Breakdown</span>
                                                <div className="space-y-3">
                                                    {Object.entries(result.probabilities).map(([cls, prob]) => (
                                                        <div key={cls}>
                                                            <div className="flex justify-between text-xs mb-1">
                                                                <span className="text-slate-300">{cls}</span>
                                                                <span className="text-slate-400">{prob}%</span>
                                                            </div>
                                                            <div className="w-full bg-slate-800 rounded-full h-2">
                                                                <div className="bg-indigo-500 h-2 rounded-full transition-all duration-500" style={{ width: `${prob}%` }}></div>
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="text-center py-12 text-slate-500">
                                            Submit feature parameters to view classification probabilities.
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    </div>
                );
            }

            ReactDOM.createRoot(document.getElementById('root')).render(<App />);
        </script>
    </body>
    </html>
    """
