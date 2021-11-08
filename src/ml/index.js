import QuestionAnswerer from './qa.js';
// import FaceDetector from './face.js';


const QA_MODEL_PATH = '/static/models/albert/model.json';
const QA_VOCAB_PATH = '/static/models/albert/vocab.json';
const BLAZEFACE_MODEL_PATH = '/static/models/blazeface/model.json';
const FACEMESH_MODEL_PATH = '/static/models/facemesh/model.json';
const IRIS_MODEL_PATH = '/static/models/iris/model.json';


const QA_THRESHOLD = 10.0;
const FACE_DETECTOR_THRESHOLD = 0.9;


const questionAnswererPromise = QuestionAnswerer.fromFiles(QA_MODEL_PATH, QA_VOCAB_PATH);
// const faceDetectorPromise = FaceDetector.fromConfig({
//     detectorModelUrl: BLAZEFACE_MODEL_PATH,
//     modelUrl: FACEMESH_MODEL_PATH,
//     irisModelUrl: IRIS_MODEL_PATH,
//     detectionConfidence: FACE_DETECTOR_THRESHOLD,
//     maxFaces: 1,
// });


export async function stopCamera() {
    const faceDetector = await faceDetectorPromise;
    await faceDetector.stopCamera();
}


export async function startCamera() {
    const faceDetector = await faceDetectorPromise;
    return await faceDetector.setupCamera();
}


export async function getAnswer(question, context, threshold = QA_THRESHOLD) {
    const questionAnswerer = await questionAnswererPromise;
    const answer = await questionAnswerer.answer(question, context, threshold);
    return answer;
}


export async function getFaces() {
    const faceDetector = await faceDetectorPromise;
    const faces = await faceDetector.detect();
    return faces;
}
