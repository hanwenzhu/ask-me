// import * as tf from './tfjs-imports.js';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';


export default class FaceDetector {
    constructor(model) {
        this.model = model;
        this.videoPromise = null;
    }

    static async fromConfig(packageConfig) {
        const modelPromise = faceLandmarksDetection.load(
            faceLandmarksDetection.SupportedPackages.mediapipeFacemesh,
            packageConfig
        );
        const model = await modelPromise;
        return new this(model);
    }

    async setupCamera() {
        if (this.videoPromise != null) {
            return this.videoPromise;
        }

        const video = document.createElement('video');
        const stream = await navigator.mediaDevices.getUserMedia({
            'audio': false,
            'video': {
                facingMode: 'user',
            },
        });
        video.srcObject = stream;

        const videoPromise = new Promise(resolve => {
            video.onloadedmetadata = () => {
                video.play();
                resolve(video);
            };
        });
        this.videoPromise = videoPromise;
        return videoPromise;
    }

    async stopCamera() {
        if (this.videoPromise == null) {
            return;
        }

        const video = await this.setupCamera();
        const stream = video.srcObject;

        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        this.videoPromise = null;
    }

    async detect() {
        const video = await this.setupCamera();
        const predictions = await this.model.estimateFaces({
            input: video,
        });
        return predictions;
    }
}
