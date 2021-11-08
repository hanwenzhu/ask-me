# ask-me
Question answering on the web with TF.js.

Try demo at [thomaszhu.cn/ask-me](https://thomaszhu.cn/ask-me).

All computing is done at the client-side JS, so the server does not need to perform any computation or collect any information. This webpage runs an ALBERT fine-tuned on SQuAD, achieving markedly better precision & a bit smaller model size than MobileBERT from the official TF.js Q&A demo. I have to rant that porting ALBERT to TF.js is a nightmare because of the weight-sharing mechanism not picked up by the converter.

I'm currently experimenting on webcam-based iris location detection, so the webpage knows where you're looking at. They're just tests and nothing will come out of it.
