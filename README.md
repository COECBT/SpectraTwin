# SpectraTwin

A comprehensive spectral analysis and machine learning pipeline application built with Streamlit. SpectraTwin provides tools for experimental design, data visualization, preprocessing, model training, neural network building, and real-time analysis of spectral data.

## Features

- **Experimental Design**: Plan and design spectral experiments with statistical rigor
- **Data Visualization**: Interactive visualization of spectral data and analysis results
- **Data Preprocessing**: Advanced preprocessing techniques for spectral data
- **Model Training**: Train machine learning models (PCA, MPLS, OPLS) on spectral data
- **Neural Network Builder**: Design and train custom neural networks
- **Control Charts**: Statistical process control and quality monitoring
- **Full Pipeline**: End-to-end automated analysis workflow
- **Model Prediction**: Make predictions using trained models
- **One-Click Pipeline**: Streamlined single-step analysis
- **Real-Time Transfer**: Live spectral data transfer and analysis
- **HPLC Integration**: High-Performance Liquid Chromatography data analysis

## Project Structure

```
spectraTwin/
├── Home.py                          # Main Streamlit app
├── pages/                           # Streamlit multi-page app pages
│   ├── 00_Experimental design.py
│   ├── 01_Data_Visualization.py
│   ├── 02_Preprocessing.py
│   ├── 03_Model_Training.py
│   ├── 04_Neural_Network_Builder.py
│   ├── 05_Control chart.py
│   ├── 06_Full_Pipeline.py
│   ├── 07_Model_prediction.py
│   ├── 08_One_Click_Pipeline.py
│   ├── 09_Real_Time_Transfer.py
│   └── 10_HPLC.py
├── Logo/                            # Application logos and assets
├── Core Modules:
│   ├── pca.py                       # Principal Component Analysis
│   ├── mpls.py                      # Modified Partial Least Squares
│   ├── opls.py                      # Orthogonal Partial Least Squares
│   ├── hplc.py                      # HPLC data processing
│   ├── FFT.py                       # Fast Fourier Transform
│   ├── preprocess.py                # Data preprocessing utilities
│   ├── data_augmentation.py         # Data augmentation techniques
│   ├── evaluate_design.py           # Experimental design evaluation
│   ├── midel.py                     # Model utilities
│   └── chatbot.py                   # AI chatbot integration
├── Streamlit Pipelines:
│   ├── streamlit_fast_pipeline.py
│   ├── streamlit_full_pipeline.py
│   └── streamlit_optimized_pipeline.py
├── realtime_client.py               # Real-time data client
└── Api.txt                          # API configuration file
```

## Prerequisites

- Python 3.8+
- pip package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Naveen-567/SpectrTwin.git
cd SpectraTwin
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. **Set up Groq API Key** (IMPORTANT):
   - Visit [Groq Console](https://console.groq.com) and create a free account
   - Generate an API key from your account dashboard
   - Create or update the `Api.txt` file in the project root:
     ```
     GroQ_API_key = your_api_key_here
     ```
   - Replace `your_api_key_here` with your actual Groq API key
   - **⚠️ DO NOT commit this file with your API key to version control**

## Running the Application

### Locally

#### Main Application
```bash
streamlit run Home.py
```

#### Fast Pipeline
```bash
streamlit run streamlit_fast_pipeline.py
```

#### Full Pipeline
```bash
streamlit run streamlit_full_pipeline.py
```

#### Optimized Pipeline
```bash
streamlit run streamlit_optimized_pipeline.py
```

#### Real-Time Data Transfer (LOCAL ONLY)
**⚠️ DO NOT use through Streamlit Cloud - must be run locally**

**Server Setup** (on main SpectraTwin system):
```bash
streamlit run Home.py
# Then navigate to page "09_Real_Time_Transfer"
# Click "▶️ Start Server" button
# Copy the WebSocket URL shown on the page
```

**Client Setup** (on data source/instrument system):
```bash
python realtime_client.py
# Paste the WebSocket URL and click Connect
```

**See "Real-Time Features" section above for complete setup guide.**

The application will open in your default web browser at `http://localhost:8501`

### Deploy on Streamlit Cloud

**Note:** Most features work on Streamlit Cloud, **EXCEPT** the real-time data transfer feature which requires local network access and persistent WebSocket connections.

**For real-time features:** Use the local setup (see "Running the Application" section above).

**Follow the complete deployment guide:** [DEPLOYMENT.md](DEPLOYMENT.md)

**Quick Summary:**
1. Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in with GitHub
2. Click "New app" and select this repository
3. Set your Groq API key in app settings → Secrets:
   ```toml
   GROQ_API_KEY = "your_api_key_here"
   ```
4. Deploy!

**Common Issues:**
- If chatbot errors occur, it will gracefully disable (app still works)
- Ensure `GROQ_API_KEY` is set in Streamlit Cloud Secrets
- Check deployment logs if issues persist

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed troubleshooting.

## Usage

1. **Experimental Design**: Start by designing your experiment with proper statistical parameters
2. **Data Preparation**: Preprocess your spectral data using the Preprocessing module
3. **Model Training**: Train appropriate models based on your data characteristics
4. **Analysis**: Run predictions and analyze results using trained models
5. **Quality Control**: Monitor results using control charts

## Configuration

### API Setup
Edit `Api.txt` to configure your Groq API credentials:
```
GroQ_API_key = your_groq_api_key
```

## Dependencies

Key dependencies include:
- streamlit
- numpy
- pandas
- scikit-learn
- groq
- plotly
- scipy

See full list in `requirements.txt`

## Real-Time Features

### ⚠️ IMPORTANT: Real-Time Transfer is for LOCAL use only

The **real-time data transfer feature** is designed to run on your **local system or internal network**, NOT through Streamlit Cloud. 

**Why?**
- Requires persistent network connections and background processes
- Streamlit Cloud sessions timeout and restart periodically
- File system access and direct network access not available on Cloud
- WebSocket connections need to remain open indefinitely

---

### Real-Time Data Transfer Setup

#### Architecture Overview:
```
┌──────────────────────────────────────┐
│  INSTRUMENT/DATA SOURCE SYSTEM       │
│  (Local PC connected to device)      │
│                                      │
│  ├─ Run: realtime_client.py          │
│  ├─ Connected to: FTIR/Raman/etc.    │
│  └─ Sends data to SpectraTwin server │
└─────────────────┬────────────────────┘
                  │ 
         Network Connection
         (LAN or Internet)
                  │
                  ↓
┌──────────────────────────────────────┐
│  SPECTRATWIN SERVER (LOCAL or SELF-HOSTED) │
│                                      │
│  ├─ Run: streamlit run Home.py       │
│  ├─ Open: Page 09_Real_Time_Transfer │
│  ├─ Click: "▶️ Start Server"          │
│  └─ View: Live predictions & data    │
└──────────────────────────────────────┘
```

---

### Step-by-Step Setup

#### 1. On the SpectraTwin Server System (Main Application):

```bash
# Navigate to project directory
cd /path/to/SpectraTwin

# Install dependencies
pip install -r requirements.txt

# Start the main app
streamlit run Home.py
```

Then:
- Open `http://localhost:8501` in your browser
- Navigate to **"09_Real_Time_Transfer"** page
- Enter the folder path to monitor (e.g., `C:\Spectra_Data`)
- **(Optional)** Upload a trained model for real-time predictions
- Click **"▶️ Start Server"** button
- **Copy the WebSocket URL** shown on the page (e.g., `ws://192.168.1.100:8765`)

#### 2. On the Data Source/Instrument System:

```bash
# Navigate to project directory
cd /path/to/SpectraTwin

# Install dependencies (same as server)
pip install -r requirements.txt

# Run the real-time client
python realtime_client.py
```

Then:
- A GUI window will open showing the client interface
- **Paste the WebSocket URL** from Step 1 in the input field
- Click **"Connect"** button
- Wait for the status to show "✅ Connected"
- Drop spectra files (`.csv`, `.txt`, `.xlsx`) in the monitored folder
- Watch real-time predictions appear in the client!

---

### Real-Time Client Features

**`realtime_client.py` provides:**
- Persistent network connection to SpectraTwin server
- Real-time spectral data visualization
- Live prediction display
- Data logging to file (CSV or TXT)
- Auto-save functionality
- Connection status monitoring

**Usage Example:**
```bash
# Terminal on data source system
python realtime_client.py
```

---

### Deployment Options

| Option | Use Case | Notes |
|--------|----------|-------|
| **Local (Same PC)** | Testing, small setup | Simplest, `ws://127.0.0.1:8765` |
| **Local Network** | Lab setup, multiple instruments | Same network, `ws://192.168.x.x:8765` |
| **Self-Hosted Server** | Production use, multiple clients | VPS or dedicated server |
| **Streamlit Cloud** | ❌ NOT SUPPORTED | Session timeout, no persistent WebSocket |

---

### Network Configuration

#### For Local Network Access:
1. Find server IP: `ipconfig` (Windows) or `ifconfig` (Linux/Mac)
2. Use in client: `ws://SERVER_IP:8765` (replace SERVER_IP)
3. Ensure firewall allows port 8765

#### For Remote Access (across internet):
1. Set up port forwarding on router
2. Use: `ws://YOUR_PUBLIC_IP:8765`
3. Consider using `wss://` (WebSocket Secure) for security

---

### Troubleshooting Real-Time Connection

**Connection Refused?**
- [ ] Server is started on SpectraTwin page (look for green checkmark)
- [ ] Server and client are on same/accessible network
- [ ] Port 8765 is not blocked by firewall
- [ ] Correct URL is pasted in client (`ws://...` not `http://...`)

**No Data Received?**
- [ ] Server folder exists and is accessible
- [ ] File being dropped has correct extension (`.csv`, `.txt`, `.xlsx`, `.spa`)
- [ ] Model file is loaded (if predictions expected)

**Connection Drops?**
- [ ] Check network stability
- [ ] Firewall not blocking WebSocket traffic
- [ ] Reconnect using the same client interface

---

### Real-Time Processing Features

**On SpectraTwin Server (09_Real_Time_Transfer page):**
- Folder monitoring with automatic file detection
- Optional model loading for real-time predictions
- WebSocket server for broadcasting to multiple clients
- Live uptime counter and connection details
- Server start/stop controls


- Ensure `realtime_client.py` is running on data source system
- Check configured IP and port match in both systems

**Issue: Groq API Key Error**
- Ensure your API key is correctly set in `Api.txt`
- Verify the key is active on the Groq console
- Check your internet connection

**Issue: Missing Dependencies**
- Run `pip install -r requirements.txt` again
- Ensure you're using the correct Python version (3.8+)

**Issue: Port Already in Use**
- The app uses port 8501 by default
- To use a different port: `streamlit run Home.py --server.port 8502`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or support, please open an issue on the GitHub repository.

---

**Note**: This application is designed for spectral data analysis in research and industrial settings. Ensure you have appropriate expertise when interpreting results for critical applications.
