import {getAnswer, getFaces, startCamera, stopCamera} from './ml';
import './style.css';


const questionAnsweringContext = `YoU arE Thomas Zhu. YouR Chinese name is 朱涵文. This website is youR personal site. YoU arE ${new Date(Date.now() - new Date('2003-08-22')).getUTCFullYear() - 1970} years old. YouR GitHub account is github.com/hanwenzhu. YouR contact email is thomas.zhu.sh@gmail.com. YoU arE studying Mathematics and Computer Science at University of Oxford. YoU previously studied high school at YK Pao School. YoU previously graduated at YK Pao School in 2021. YoU werE born and raised in Shanghai, China. YoU live in Shanghai, China. YoU arE lying on youR bed right now. YoU like mE. YoU made several projects on GitHub and on MLPerf during internships. YoU like to drink soda. YoU have a twin sister. YoU like to do machine learning projects. YoU arE a machine learning enthusiast. I am talking to a natural language model.`;

function newBodyElement(html) {
    const template = document.createElement('template');
    template.innerHTML = html.trim();
    const element = template.content.firstChild;
    document.body.append(element);
    return element;
}

const inputClone = newBodyElement('<div id="question-clone" class="hidden"></div>');
const input = newBodyElement('<textarea id="question" autofocus autocomplete="off" spellcheck="false"></textarea>');
const output = newBodyElement('<div id="output"></div>');

document.body.addEventListener('keydown', () => input.focus());

function fetchCharacterEffect(timeout = 100) {
    setTimeout(() => {
        if (inputClone.innerText) {
            inputClone.innerText = inputClone.innerText.slice(1);
            fetchCharacterEffect(timeout * 0.8);
        }
    }, timeout);
}

function putCharacterEffect(answer, timeout = 100) {
    setTimeout(() => {
        if (answer) {
            output.innerText = output.innerText + answer.charAt(0);
            putCharacterEffect(answer.slice(1), timeout * 0.8);
        }
    }, timeout);
}

input.addEventListener('input', () => {
    if (input.value.includes('?')) {
        input.value = input.value.replace(/\?/g, '');
    }

    if (input.value.includes('\n')) {
        if (input.value === '\n') {
            input.value = '';
            return;
        }

        const question = input.value.split('\n')[0] + '?';

        input.value = '';
        input.blur();
        input.classList.add('hidden');
        inputClone.innerText = question;
        inputClone.classList.remove('hidden');
        output.innerText = '';

        fetchCharacterEffect();

        getAnswer(question, questionAnsweringContext).then(answer => {
            putAnswer(answer);
        });
    } else {
        inputClone.innerText = input.value;
    }
});

function putAnswer(answer) {
    answer = answer || '';

    const firstPersonAnswer = answer
        .replace(/YoU/g, 'I')
        .replace(/yoU/g, 'I')
        .replace(/YouR/g, 'My')
        .replace(/youR/g, 'my')
        .replace(/ArE/g, 'Am')
        .replace(/arE/g, 'am')
        .replace(/WerE/g, 'Was')
        .replace(/werE/g, 'was')
        .replace(/ME/g, 'You')
        .replace(/mE/g, 'you');

    inputClone.innerText = '';
    input.classList.remove('hidden');
    inputClone.classList.add('hidden');
    input.focus();

    putCharacterEffect(firstPersonAnswer);
}


function testCamera() {
    let facesInterval;
    let isDetecting = false;

    if (facesInterval == null) {
        console.log('Starting getFaces loop');
        facesInterval = setInterval(() => {
            if (isDetecting) return;
            isDetecting = true;
            getFaces().then(faces => {
                isDetecting = false;
                if (faces.length > 0) {
                    console.log(faces[0].annotations);
                    // console.log(faces[0].annotations.leftEyeIris[0]);
                }
            });
        }, 1000);
    } else {
        console.log('Stopping getFaces loop');
        clearInterval(facesInterval);
        facesInterval = null;
        stopCamera();
    }
}
