const express = require('express');
const bodyParser = require('body-parser');
const multer = require('multer');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');
const cors = require('cors');
const fs = require('fs').promises; // Use fs.promises for asynchronous operations

const app = express();
const port = 3000;

app.use(cors());
app.use(bodyParser.json());
app.use(express.static('public')); // Serve static files from the public directory

// Initialize multer for file uploads
const upload = multer({ dest: 'uploads/' });

// Load the model once at the start
let model;
async function loadModel() {
    try {
        const modelPath = path.join(__dirname, 'public/Model', 'model.json');
        model = await tf.loadLayersModel(`file://${modelPath}`);
        console.log('Model loaded successfully');
    } catch (error) {
        console.error('Error loading model:', error);
    }
}
loadModel().catch(console.error);

// Endpoint to handle file uploads and predictions
app.post('/predict', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).send('No file uploaded.');
        }

        // Read the image file using fs.promises
        const imageBuffer = await fs.readFile(req.file.path);
        const imageTensor = tf.node.decodeImage(imageBuffer)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .expandDims();

        const prediction = await model.predict(imageTensor).data();
        const classIndices = await fetchClassIndices();

        const predictedClassIndex = tf.argMax(prediction);
        const classIndex = Array.from(predictedClassIndex.dataSync())[0];
        const className = classIndices[classIndex];

        // Clean up uploaded file using fs.promises
        await fs.unlink(req.file.path);

        res.json({
            className,
            confidence: parseFloat(prediction[classIndex] * 100).toFixed(2)
        });
    } catch (error) {
        res.status(500).send(error.message);
    }
});

// Function to fetch class indices
async function fetchClassIndices() {
    try {
        const filePath = path.join(__dirname, 'public', 'Model', 'class_indices.json');
        const data = await fs.readFile(filePath, 'utf8'); // Specify 'utf8' as encoding
        return JSON.parse(data);
    } catch (error) {
        console.error('Failed to fetch class indices:', error);
        throw error;
    }
}

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
