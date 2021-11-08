// import * as tf from './tfjs-imports.js';
import * as tf from '@tensorflow/tfjs';


class TrieNode {
    constructor() {
        this.parent = null;
        // symbol -> node map
        this.children = {};
        this.end = false;
        this.word = {token: [], score: 0, index: 0};
    }
}


class Trie {
    constructor() {
        this.root = new TrieNode();
    }

    insert(word, score, index) {
        let node = this.root;

        // Unicode-aware splitting
        const symbols = [...word];

        for (let i = 0; i < symbols.length; i++) {
            if (!node.children[symbols[i]]) {
                node.children[symbols[i]] = new TrieNode();
                node.children[symbols[i]].parent = node;
                node.children[symbols[i]].word.token = node.word.token.concat(symbols[i]);
            }

            node = node.children[symbols[i]];
            if (i === symbols.length - 1) {
                node.end = true;
                node.word.score = score;
                node.word.index = index;
            }
        }
    }

    prefixSearch(query) {
        const output = [];
        let node = this.root.children[query[0]];

        for (let i = 0; i < query.length && node; i++){
            if (node.end) {
                output.push(node.word);
            }
            node = node.children[query[i + 1]];
        }

        return output;
    }
}


class Tokenizer {
    constructor(vocabulary,
                reservedSymbolsCount = 5,
                padId = 0, unkId = 1, clsId = 2, sepId = 3, maskId = 4,
                separator = '\u2581',
                lowerCase = true, maxSeqLen = 512) {
        this.vocabulary = vocabulary;
        this.reservedSymbolsCount = reservedSymbolsCount;
        this.padId = padId;
        this.unkId = unkId;
        this.clsId = clsId;
        this.sepId = sepId;
        this.maskId = maskId;
        this.separator = separator;
        this.lowerCase = lowerCase;
        this.maxSeqLen = maxSeqLen;

        this.trie = new Trie();

        for (let i = this.reservedSymbolsCount; i < this.vocabulary.length; i++) {
            this.trie.insert(this.vocabulary[i][0], this.vocabulary[i][1], i);
        }
    }

    static async fromVocabFile(path, ...args) {
        const response = await fetch(path);
        const vocabulary = await response.json();
        return new this(vocabulary, ...args);
    }

    encode(input) {
        // NFKD normalization & other reductions (semantically equivalent, length-changing)
        const normalized = input
            .trim()
            .replace(/\s+/g, ' ')
            .replace(/``|''/g, '"')
            .normalize('NFKD')
            .replace(/[\u0300-\u036f]/g, '');

        // Lower case & whitespace separation (semantically inequivalent, length-preserving)
        const processed = (this.lowerCase ? normalized.toLowerCase() : normalized)
            .replace(/^| /g, this.separator);

        console.assert(normalized.length === processed.length - 1);

        // Unicode-aware splitting
        const symbols = [...processed];

        const nodes = new Array(symbols.length + 1).fill(null).map(x => ({}));
        const words = new Array(symbols.length + 1).fill(0);
        const best = new Array(symbols.length + 1).fill(0);

        // Lattice construction
        for (let i = 0; i < symbols.length; i++) {
            const matches = this.trie.prefixSearch(symbols.slice(i));

            if (matches.length === 0) {
                matches.push({token: [symbols[i]], score: 0, index: this.unkId});
            }

            for (const piece of matches) {
                const endPos = piece.token.length;

                if (!(i in nodes[i + endPos])) {
                    nodes[i + endPos][i] = [];
                }
                nodes[i + endPos][i].push(piece);
            }
        }

        for (let endPos = 0; endPos <= symbols.length; endPos++) {
            for (const startPos in nodes[endPos]) {
                for (const word of nodes[endPos][startPos]) {
                    const score = word.score + best[endPos - word.token.length];

                    if (best[endPos] === 0 || score >= best[endPos]) {
                        best[endPos] = score;
                        words[endPos] = word;
                    }
                }
            }
        }

        // Backward pass
        const tokens = [];
        const origs = [];
        for (let endPos = words.length - 1; endPos > 0;) {
            const wordLength = words[endPos].index < this.reservedSymbolsCount
                // Special symbols take 1 byte
                ? 1
                // Unicode-aware length
                : [...words[endPos].token].length;

            // Subtract 1 leading separator
            const orig = normalized.slice(endPos > wordLength ? endPos - wordLength - 1 : 0, endPos - 1);

            if (tokens[0] === this.unkId && words[endPos].index === this.unkId) {
                // Merge consecutive unknowns
                origs[0] = orig + origs[0];
            } else {
                // Push new word to tokens
                tokens.unshift(words[endPos].index);
                origs.unshift(orig);
            }

            endPos -= wordLength;
        }

        return {tokens: tokens, origs: origs};
    }

    pack(question, context, pad = false) {
        const {tokens: questionTokens, origs: questionOrigs} = this.encode(question);
        const {tokens: contextTokens, origs: contextOrigs} = this.encode(context);

        const maskLength = 1 + questionTokens.length + 1 + contextTokens.length + 1;

        if (maskLength > this.maxSeqLen) {
            throw new Error(`Token length greater than model maximum sequence length (${maskLength} > ${this.maxSeqLen})`);
        }
        const padLength = pad && maskLength <= this.maxSeqLen ? this.maxSeqLen - maskLength : 0;

        const tokens = [
            this.clsId,
            ...questionTokens,
            this.sepId,
            ...contextTokens,
            this.sepId,
            ...new Array(padLength).fill(this.padId)
        ];
        const tokenTypes = [
            ...new Array(1 + questionTokens.length + 1).fill(0),
            ...new Array(contextTokens.length + 1).fill(1),
            ...new Array(padLength).fill(0)
        ];
        const mask = [
            ...new Array(maskLength).fill(1),
            ...new Array(padLength).fill(0)
        ];
        const origs = [
            '',
            ...questionOrigs,
            '',
            ...contextOrigs,
            '',
            ...new Array(padLength).fill('')
        ];

        return {tokens: tokens, tokenTypes: tokenTypes, mask: mask, origs: origs};
    }
}


function getActivation(activation) {
    if (typeof activation === 'string') {
        return tf.layers.activation({activation: activation});
    } else if (activation != null) {
        return activation;
    } else {
        return tf.layers.activation({activation: 'linear'});
    }
}

function getInitializer(initializer) {
    return getTensorflowObject(initializer, tf.initializers);
}

function getRegularizer(regularizer) {
    return getTensorflowObject(regularizer, tf.regularizers);
}

function getConstraint(constraint) {
    return getTensorflowObject(constraint, tf.constraints);
}

function getTensorflowObject(object, nameSpace) {
    if (object == null) {
        return;
    } else if (typeof object === 'string') {
        return nameSpace[object]();
    } else if (object.className && object.config) {
        return nameSpace[object.className[0].toLowerCase() + object.className.slice(1)](object.config);
    } else {
        return object;
    }
}


function serialize(instance) {
    if (instance.constructor.className === 'Activation') {
        return instance.activation.getClassName();
    } else {
        return {
            className: instance.getClassName(),
            config: instance.getConfig()
        };
    }
}


tf.layers.Layer.prototype.addSubLayer = function(subLayer, inputShape) {
    subLayer.build(inputShape);
    subLayer.built = true;

    subLayer.weights.forEach(weight => {
        if (!weight.nameModified) {
            weight.nameModified = true;
            const scopedName = weight.name.split('/');
            weight.name = `${this.name}/${subLayer.name}/${scopedName[scopedName.length - 1]}`;
            const scopedOriginalName = weight.originalName.split('/');
            weight.originalName = `${this.name}/${subLayer.name}/${scopedOriginalName[scopedOriginalName.length - 1]}`;
        } else {
            weight.name = `${this.name}/${weight.name}`;
            weight.originalName = `${this.name}/${weight.originalName}`;
        }
        if (this._addedWeightNames.indexOf(weight.name) !== -1) {
          throw new Error(`Duplicate weight name ${weight.name} for layer ${this.name}`);
        }
        this._addedWeightNames.push(weight.name);
    });
    subLayer.trainableWeights.forEach(weight => {
        this._trainableWeights.push(weight);
    });
    subLayer.nonTrainableWeights.forEach(weight => {
        this._nonTrainableWeights.push(weight);
    });

    subLayer.losses.forEach(loss => this.addLoss(loss));

    return subLayer;
}


function einsum(equation, x1, x2) {
    if (['abc,cde->abde', 'abc,cd->abd', '...x,xy->...y'].includes(equation)) {
        return tensordot(x1, x2, 1);
    } else if (equation == 'abcd,cde->abe') {
        return tensordot(x1, x2, 2);
    } else if (equation == 'aecd,abcd->acbe') {
        x1 = tf.transpose(x1, [0, 2, 1, 3]);
        x2 = tf.transpose(x2, [0, 2, 3, 1]);
        const y = tf.matMul(x1, x2);
        return tf.transpose(y, [0, 1, 3, 2]);
    } else if (equation == 'acbe,aecd->abcd') {
        x2 = tf.transpose(x2, [0, 2, 1, 3]);
        const y = tf.matMul(x1, x2);
        return tf.transpose(y, [0, 2, 1, 3]);
    } else {
        throw new Error(`Ad-hoc implementation of einsum does not support ${equation}`);
    }
}


function tensordot(x1, x2, nDims) {
    const shape1 = x1.shape;
    const shape2 = x2.shape;
    x1 = tf.reshape(x1, [-1, tf.util.sizeFromShape(shape1.slice(-nDims))]);
    x2 = tf.reshape(x2, [tf.util.sizeFromShape(shape2.slice(0, nDims)), -1]);
    const y = tf.dot(x1, x2);
    return tf.reshape(y, [...shape1.slice(0, -nDims), ...shape2.slice(nDims)]);
}


class OnDeviceEmbedding extends tf.layers.Layer {

    constructor(config) {
        super(config);
        this._vocabSize = config.vocabSize;
        this._embeddingWidth = config.embeddingWidth;
        this._initializer = getInitializer(config.initializer || 'glorotUniform');
        this._useOneHot = config.useOneHot || false;
        this._scaleFactor = config.scaleFactor || null;
    }

    getConfig() {
        const config = {
            vocabSize: this._vocabSize,
            embeddingWidth: this._embeddingWidth,
            initializer: serialize(this._initializer),
            useOneHot: this._useOneHot,
            scaleFactor: this._scaleFactor,
        };
        const baseConfig = super.getConfig();
        return {...baseConfig, ...config};
    }

    build(inputShape) {
         this.embeddings = this.addWeight(
            'embeddings',
            [this._vocabSize, this._embeddingWidth],
            'float32',
            this._initializer,
        );

        super.build(inputShape);
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs;
            const flatInputs = tf.reshape(inputs, [-1]);
            let embeddings;
            if (this._useOneHot) {
                const dtype = this.dtype || 'float32';
                const oneHotData = tf.cast(tf.oneHot(flatInputs, this._vocabSize), dtype);
                embeddings = tf.matMul(oneHotData, this.embeddings.read());
            } else {
                embeddings = tf.gather(this.embeddings.read(), flatInputs);
            }
            embeddings = tf.reshape(
                embeddings,
                [...inputs.shape, this._embeddingWidth]
            );
            if (this._scaleFactor) {
                embeddings = tf.mul(embeddings, this._scaleFactor);
            }
            return embeddings;
        });
    }

    get vocabSize() {
        return this._vocabSize;
    }

    get embeddingWidth() {
        return this._embeddingWidth;
    }

    static get className() {
        return 'kerasNlp>OnDeviceEmbedding';
    }

    computeOutputShape(inputShape) {
        return [...inputShape, this._embeddingWidth];
    }
}

tf.serialization.registerClass(OnDeviceEmbedding);


class PositionEmbedding extends tf.layers.Layer {

    constructor(config) {
        super(config);
        if (config.maxLength == null) {
            throw new Error('`maxLength` must be an Integer, not `null` or `undefined`.');
        }
        this._maxLength = config.maxLength;
        this._initializer = getInitializer(config.initializer || 'glorotUniform');
    }

    getConfig() {
        const config = {
            maxLength: this._maxLength,
            initializer: serialize(this._initializer),
        };
        const baseConfig = super.getConfig();
        return {...baseConfig, ...config};
    }

    build(inputShape) {
        const dimensionList = inputShape;

        if (dimensionList.length !== 3) {
            throw new Error(`PositionEmbedding expects a 3-dimensional input tensor of shape [batch, sequence, width], got ${inputShape}`);
        }

        const seqLength = dimensionList[1];
        const width = dimensionList[2];

        const weightSequenceLength = this._maxLength != null ? this._maxLength : seqLength;
        this._positionEmbeddings = this.addWeight(
            'embeddings',
            [weightSequenceLength, width],
            undefined,
            this._initializer
        );

        super.build(inputShape);
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs;
            const inputShape = inputs.shape;
            const positionEmbeddings = this._positionEmbeddings.read().slice([0, 0], [inputShape[1], -1]);
            return tf.broadcastTo(positionEmbeddings, inputShape);
        });
    }

    static get className() {
        return 'kerasNlp>PositionEmbedding';
    }
}

tf.serialization.registerClass(PositionEmbedding);


class SelfAttentionMask extends tf.layers.Layer {

    call([inputs, toMask]) {
        return tf.tidy(() => {
            const fromShape = inputs.shape;
            const batchSize = fromShape[0];
            const fromSeqLength = fromShape[1];

            const toShape = toMask.shape;
            const toSeqLength = toShape[1];

            toMask = tf.cast(
                tf.reshape(toMask, [batchSize, 1, toSeqLength]),
                inputs.dtype
            );

            const broadcastOnes = tf.ones([batchSize, fromSeqLength, 1], inputs.dtype);

            const mask = tf.mul(broadcastOnes, toMask);

            return mask;
        });
    }

    static get className() {
        return 'kerasNlp>SelfAttentionMask';
    }

    computeOutputShape([fromShape, toShape]) {
        const batchSize = fromShape[0];
        const fromSeqLength = fromShape[1];
        const toSeqLength = toShape[1];
        return [batchSize, fromSeqLength, toSeqLength];
    }
}

tf.serialization.registerClass(SelfAttentionMask);


const _CHR_IDX = 'abcdefghijklmnopqrstuvwxyz';


function _buildAttentionEquation(rank, attnAxes) {
    const targetNotation = _CHR_IDX.slice(0, rank);
    const batchDims = [...new Array(rank).keys()].filter(a => !attnAxes.includes(a) && a !== rank - 1);
    let letterOffset = rank;
    let sourceNotation = '';
    for (let i = 0; i < rank; i++) {
        if (batchDims.includes(i) || i === rank - 1) {
            sourceNotation += targetNotation[i];
        } else {
            sourceNotation += _CHR_IDX[letterOffset];
            letterOffset += 1;
        }
    }

    const productNotation = [
        ...batchDims.map(i => targetNotation[i]),
        ...attnAxes.map(i => targetNotation[i]),
        ...attnAxes.map(i => sourceNotation[i])
    ].join('');
    const dotProductEquation = `${sourceNotation},${targetNotation}->${productNotation}`;
    const attnScoresRank = productNotation.length;
    const combineEquation = `${productNotation},${sourceNotation}->${targetNotation}`;
    return [dotProductEquation, combineEquation, attnScoresRank];
}


function _buildProjEquation(freeDims, boundDims, outputDims) {
    let inputStr = '';
    let kernelStr = '';
    let outputStr = '';
    let biasAxes = '';
    let letterOffset = 0;
    for (let i = 0; i < freeDims; i++) {
        const char = _CHR_IDX[i + letterOffset];
        inputStr += char;
        outputStr += char;
    }

    letterOffset += freeDims;
    for (let i = 0; i < boundDims; i++) {
        const char = _CHR_IDX[i + letterOffset];
        inputStr += char;
        kernelStr += char;
    }

    letterOffset += boundDims
    for (let i = 0; i < outputDims; i++) {
        const char = _CHR_IDX[i + letterOffset];
        kernelStr += char;
        outputStr += char;
        biasAxes += char;
    }
    const equation = `${inputStr},${kernelStr}->${outputStr}`;

    return [equation, biasAxes, outputStr.length];
}


function _getOutputShape(outputRank, knownLastDims) {
    return [...new Array(outputRank - knownLastDims.length).fill(null), ...knownLastDims];
}


class MultiHeadAttention extends tf.layers.Layer {

    constructor(config) {
        super(config);
        this._numHeads = config.numHeads;
        this._keyDim = config.keyDim;
        this._valueDim = config.valueDim || config.keyDim;
        this._dropout = config.dropout || 0.0;
        this._useBias = config.useBias || true;
        this._outputShape = config.outputShape || null;
        this._kernelInitializer = getInitializer(config.kernelInitializer || 'glorotUniform');
        this._biasInitializer = getInitializer(config.biasInitializer || 'zeros');
        this._kernelRegularizer = getRegularizer(config.kernelRegularizer);
        this._biasRegularizer = getRegularizer(config.biasRegularizer);
        this._kernelConstraint = getConstraint(config.kernelConstraint);
        this._biasConstraint = getConstraint(config.biasConstraint);
        this._attentionAxes = config.attentionAxes != null && !Array.isArray(config.attentionAxes) ? [config.attentionAxes] : config.attentionAxes;
    }

    getConfig() {
        const config = {
            numHeads: this._numHeads,
            keyDim: this._keyDim,
            valueDim: this._valueDim,
            dropout: this._dropout,
            useBias: this._useBias,
            outputShape: this._outputShape,
            attentionAxes: this._attentionAxes,
            kernelInitializer: serialize(this._kernelInitializer),
            biasInitializer: serialize(this._biasInitializer),
            kernelRegularizer: serialize(this._kernelRegularizer),
            biasRegularizer: serialize(this._biasRegularizer),
            activityRegularizer: serialize(this._activityRegularizer),
            kernelConstraint: serialize(this._kernelConstraint),
            biasConstraint: serialize(this._biasConstraint)
        }
        const baseConfig = super.getConfig();
        return {...baseConfig, ...config};
    }

    // Original: _buildFromSignature, called by call(). Changed to build()
    build([queryShape, valueShape, keyShape]) {
        keyShape = keyShape == null ? valueShape : keyShape;

        const commonKwargs = {
            kernelInitializer: this._kernelInitializer,
            biasInitializer: this._biasInitializer,
            kernelRegularizer: this._kernelRegularizer,
            biasRegularizer: this._biasRegularizer,
            activityRegularizer: this._activityRegularizer,
            kernelConstraint: this._kernelConstraint,
            biasConstraint: this._biasConstraint
        };
        const freeDims = queryShape.length - 1;
        let [einsumEquation, biasAxes, outputRank] = _buildProjEquation(freeDims, 1, 2);
        // Manually build sub-layers for TF.js; ditto; see utils.js
        this._queryDense = this.addSubLayer(new EinsumDense({
            equation: einsumEquation,
            outputShape: _getOutputShape(outputRank - 1, [this._numHeads, this._keyDim]),
            biasAxes: this._useBias ? biasAxes : null,
            name: 'query',
            ...commonKwargs
        }), queryShape);
        [einsumEquation, biasAxes, outputRank] = _buildProjEquation(keyShape.length - 1, 1, 2);
        this._keyDense = this.addSubLayer(new EinsumDense({
            equation: einsumEquation,
            outputShape: _getOutputShape(outputRank - 1, [this._numHeads, this._keyDim]),
            biasAxes: this._useBias ? biasAxes : null,
            name: 'key',
            ...commonKwargs
        }), keyShape);
        [einsumEquation, biasAxes, outputRank] = _buildProjEquation(valueShape.length - 1, 1, 2);
        this._valueDense = this.addSubLayer(new EinsumDense({
            equation: einsumEquation,
            outputShape: _getOutputShape(outputRank - 1, [this._numHeads, this._valueDim]),
            biasAxes: this._useBias ? biasAxes : null,
            name: 'value',
            ...commonKwargs
        }), valueShape);

        this._buildAttention(outputRank, queryShape, keyShape);
        let outputShape;
        if (this._outputShape) {
            if (!Array.isArray(this._outputShape)) {
                outputShape = [this._outputShape];
            } else {
                outputShape = this._outputShape;
            }
        } else {
            outputShape = queryShape.slice(-1);
        }
        [einsumEquation, biasAxes, outputRank] = _buildProjEquation(freeDims, 2, outputShape.length);
        this._outputDense = this.addSubLayer(new EinsumDense({
            equation: einsumEquation,
            outputShape: _getOutputShape(outputRank - 1, outputShape),
            biasAxes: this._useBias ? biasAxes : null,
            name: 'attention_output',
            ...commonKwargs
        }), [...queryShape.slice(0, -1), this._numHeads, this._keyDim]);
    }

    _buildAttention(rank, queryShape, keyShape) {
        if (this._attentionAxes == null) {
            this._attentionAxes = [...new Array(rank - 3).keys()].map(a => a + 1);
        } else {
            this._attentionAxes = [...this._attentionAxes];
        }
        let attnScoresRank;
        [this._dotProductEquation, this._combineEquation, attnScoresRank] = _buildAttentionEquation(rank, this._attentionAxes);
        const normAxes = [...new Array(this._attentionAxes.length).keys()].map(a => a + attnScoresRank - this._attentionAxes.length);
        const batchAxes = [...new Array(rank).keys()].filter(a => !this._attentionAxes.includes(a) && a !== rank - 1);
        const attentionScoresShape = [
            ...batchAxes.map(a => queryShape[a]),
            this._numHeads,
            ...this._attentionAxes.map(a => queryShape[a]),
            ...this._attentionAxes.map(a => keyShape[a])
        ];
        // Multiple-axes softmax not supported in JS yet
        const [normAxis] = normAxes;
        this._softmax = this.addSubLayer(tf.layers.softmax({
            axis: normAxis
        }), attentionScoresShape);
        this._dropoutLayer = this.addSubLayer(tf.layers.dropout({
            rate: this._dropout
        }), attentionScoresShape);
    }

    _maskedSoftmax(attentionScores, attentionMask = null) {
        if (attentionMask != null) {
            const maskExpansionAxes = [-this._attentionAxes.length * 2 - 1];
            for (let i = 0; i < attentionScores.shape.length - attentionMask.shape.length; i++) {
                attentionMask = tf.expandDims(attentionMask, maskExpansionAxes);
            }
        }
        // Softmax mask argument not supported in JS
        // return this._softmax.apply(attentionScores, attentionMask);
        const adder = tf.mul(
            tf.sub(1.0, tf.cast(attentionMask, attentionScores.dtype)),
            attentionScores.dtype === 'float16' ? 65500.0 : -1e9
        );
        attentionScores = tf.add(attentionScores, adder);
        return this._softmax.apply(attentionScores);
    }

    _computeAttention(query, key, value, attentionMask = null, training = null) {
        query = tf.mul(query, 1.0 / Math.sqrt(this._keyDim));

        // TODO: use tf.einsum when TF.js supports Einsum op
        let attentionScores = einsum(this._dotProductEquation, key, query);

        attentionScores = this._maskedSoftmax(attentionScores, attentionMask);

        const attentionScoresDropout = this._dropoutLayer.apply(attentionScores, {training: training});

        // TODO: use tf.einsum when TF.js supports Einsum op
        const attentionOutput = einsum(this._combineEquation, attentionScoresDropout, value);
        return [attentionOutput, attentionScores];
    }

    call([query, value], args) {
        return tf.tidy(() => {
            let key = args.key || null;
            const attentionMask = args.attentionMask || null;
            const returnAttentionScores = args.returnAttentionScores || false;
            const training = args.training || null;

            if (key == null) {
                key = value;
            }

            query = this._queryDense.apply(query);

            key = this._keyDense.apply(key);

            value = this._valueDense.apply(value);

            let [attentionOutput, attentionScores] = this._computeAttention(query, key, value, attentionMask, training);
            attentionOutput = this._outputDense.apply(attentionOutput);

            if (returnAttentionScores) {
                return [attentionOutput, attentionScores];
            }
            return attentionOutput;
        });
    }

    computeOutputShape([queryShape, valueShape]) {
        if (this._outputShape) {
            return [...queryShape.slice(0, -1), ...this._outputShape];
        } else {
            return queryShape;
        }
    }
}


class TransformerEncoderBlock extends tf.layers.Layer {

    constructor(config) {
        super(config);

        this._numHeads = config.numAttentionHeads;
        this._innerDim = config.innerDim;
        this._innerActivation = config.innerActivation;
        this._attentionDropout = config.attentionDropout || 0.0;
        this._outputDropout = config.outputDropout || 0.0;
        this._outputRange = config.outputRange || null;
        this._kernelInitializer = getInitializer(config.kernelInitializer || 'glorotUniform');
        this._biasInitializer = getInitializer(config.biasInitializer || 'zeros');
        this._kernelRegularizer = getRegularizer(config.kernelRegularizer);
        this._biasRegularizer = getRegularizer(config.biasRegularizer);
        this._activityRegularizer = getRegularizer(config.activityRegularizer);
        this._kernelConstraint = getConstraint(config.kernelConstraint);
        this._biasConstraint = getConstraint(config.biasConstraint);
        this._useBias = config.useBias || true;
        this._normFirst = config.normFirst || false;
        this._normEpsilon = config.normEpsilon || 1e-12;
        this._innerDropout = config.innerDropout || 0.0;
        this._attentionInitializer = getInitializer(config.attentionInitializer || this._kernelInitializer);
    }

    build(inputShape) {
        if (!Array.isArray(inputShape)) {
            throw new Error(`The type of input shape argument is not supported, got: ${inputShape.constructor.name}`);
        }

        let inputTensorShape, keyShape, maskShape;
        if (Array.isArray(inputShape[0])) {
            if (inputShape.length === 2) {
                [inputTensorShape, maskShape] = inputShape;
            } else if (inputShape.length === 3) {
                [inputTensorShape, keyShape, maskShape] = inputShape;
            } else {
                throw new Error(`Unexpected inputs to ${this.constructor.name} with length at ${inputShape.length}`);
            }
        } else {
            inputTensorShape = inputShape;
        }

        if (inputTensorShape.length !== 3) {
            throw new Error('TransformerEncoderBlock expects a three-dimensional input of shape [batch, sequence, width].');
        }
        const hiddenSize = inputTensorShape[2];
        if (hiddenSize % this._numHeads !== 0) {
            throw new Error(`The input size (${hiddenSize}) is not a multiple of the number of attention heads (${this._numHeads})`);
        }

        const targetShape = this._outputRange ? inputTensorShape : [inputTensorShape[0], this._outputRange, inputTensorShape[2]];

        if (keyShape == null) {
            keyShape = inputTensorShape;
        }

        this._attentionHeadSize = Math.floor(hiddenSize / this._numHeads);
        const commonKwargs = {
            biasInitializer: this._biasInitializer,
            kernelRegularizer: this._kernelRegularizer,
            biasRegularizer: this._biasRegularizer,
            activityRegularizer: this._activityRegularizer,
            kernelConstraint: this._kernelConstraint,
            biasConstraint: this._biasConstraint
        };
        this._attentionLayer = this.addSubLayer(new MultiHeadAttention({
            numHeads: this._numHeads,
            keyDim: this._attentionHeadSize,
            dropout: this._attentionDropout,
            useBias: this._useBias,
            kernelInitializer: this._attentionInitializer,
            name: 'self_attention',
            ...commonKwargs
        }), [targetShape, keyShape]);
        this._attentionDropout = this.addSubLayer(tf.layers.dropout({
            rate: this._outputDropout
        }), targetShape);
        this._attentionLayerNorm = this.addSubLayer(tf.layers.layerNormalization({
            name: 'self_attention_layer_norm',
            axis: -1,
            epsilon: this._normEpsilon,
            dtype: 'float32'
        }), targetShape);
        this._intermediateDense = this.addSubLayer(new EinsumDense({
            equation: 'abc,cd->abd',
            outputShape: [null, this._innerDim],
            biasAxes: 'd',
            kernelInitializer: this._kernelInitializer,
            name: 'intermediate',
            ...commonKwargs
        }), targetShape);
        const policy = 'float32';
        const innerShape = [targetShape[0], targetShape[1], this._innerDim];
        this._intermediateActivationLayer = this.addSubLayer(tf.layers.activation({
            activation: this._innerActivation,
            dtype: policy
        }), innerShape);
        this._innerDropoutLayer = this.addSubLayer(tf.layers.dropout({
            rate: this._innerDropout
        }), innerShape);
        this._outputDense = this.addSubLayer(new EinsumDense({
            equation: 'abc,cd->abd',
            outputShape: [null, hiddenSize],
            biasAxes: 'd',
            name: 'output',
            kernelInitializer: this._kernelInitializer,
            ...commonKwargs
        }), innerShape);
        this._outputDropout = this.addSubLayer(tf.layers.dropout({
            rate: this._outputDropout
        }), targetShape);
        this._outputLayerNorm = this.addSubLayer(tf.layers.layerNormalization({
            name: 'output_layer_norm',
            axis: -1,
            epsilon: this._normEpsilon,
            dtype: 'float32'
        }), targetShape);

        super.build(inputShape)
    }

    getConfig() {
        const config = {
            numAttentionHeads: this._numHeads,
            innerDim: this._innerDim,
            innerActivation: this._innerActivation,
            outputDropout: this._outputDropout,
            attentionDropout: this._attentionDropout,
            outputRange: this._outputRange,
            kernelInitializer: serialize(this._kernelInitializer),
            biasInitializer: serialize(this._biasInitializer),
            kernelRegularizer: serialize(this._kernelRegularizer),
            biasRegularizer: serialize(this._biasRegularizer),
            activityRegularizer: serialize(this._activityRegularizer),
            kernelConstraint: serialize(this._kernelConstraint),
            biasConstraint: serialize(this._biasConstraint),
            useBias: this._useBias,
            normFirst: this._normFirst,
            normEpsilon: this._normEpsilon,
            innerDropout: this._innerDropout,
            attentionInitializer: serialize(this._attentionInitializer)
        };
        const baseConfig = super.getConfig();
        return {...baseConfig, ...config};
    }

    call(inputs) {
        return tf.tidy(() => {
            let inputTensor, keyValue, attentionMask;
            if (!Array.isArray(inputs)) {
                inputTensor = inputs;
            } if (inputs.length === 1) {
                [inputTensor] = inputs;
            } if (inputs.length === 2) {
                [inputTensor, attentionMask] = inputs;
            } else if (inputs.length === 3) {
                [inputTensor, keyValue, attentionMask] = inputs;
            } else {
                throw new Error(`Unexpected inputs to ${this.constructor.name} with length at ${inputs.length}`);
            }

            let sourceTensor, targetTensor;
            if (this._outputRange) {
                if (this._normFirst) {
                    sourceTensor = inputTensor.slice([0, 0, 0], [-1, this._outputRange, -1]);
                    inputTensor = this._attentionLayerNorm.apply(inputTensor);
                    if (keyValue != null) {
                        keyValue = this._attentionLayerNorm.apply(keyValue);
                    }
                }
                targetTensor = inputTensor.slice([0, 0, 0], [-1, this._outputRange, -1]);
                if (attentionMask != null) {
                    attentionMask = attentionMask.slice([0, 0, 0], [-1, this._outputRange, -1]);
                }
            } else {
                if (this._normFirst) {
                    sourceTensor = inputTensor;
                    inputTensor = this._attentionLayerNorm.apply(inputTensor);
                    if (keyValue != null) {
                        keyValue = this._attentionLayerNorm.apply(keyValue);
                    }
                }
                targetTensor = inputTensor;
            }

            if (keyValue == null) {
                keyValue = inputTensor;
            }
            let attentionOutput = this._attentionLayer.apply(
                [targetTensor, keyValue],
                {attentionMask: attentionMask}
            );
            attentionOutput = this._attentionDropout.apply(attentionOutput);
            if (this._normFirst) {
                attentionOutput = tf.add(sourceTensor, attentionOutput);
            } else {
                attentionOutput = this._attentionLayerNorm.apply(tf.add(targetTensor, attentionOutput));
            }
            let sourceAttentionOutput;
            if (this._normFirst) {
                sourceAttentionOutput = attentionOutput;
                attentionOutput = this._outputLayerNorm.apply(attentionOutput);
            }
            let innerOutput = this._intermediateDense.apply(attentionOutput);
            innerOutput = this._intermediateActivationLayer.apply(innerOutput);
            innerOutput = this._innerDropoutLayer.apply(innerOutput);
            let layerOutput = this._outputDense.apply(innerOutput);
            layerOutput = this._outputDropout.apply(layerOutput);

            if (this._normFirst) {
                return tf.add(sourceAttentionOutput, layerOutput);
            }

            layerOutput = tf.cast(layerOutput, 'float32');
            return this._outputLayerNorm.apply(tf.add(layerOutput, attentionOutput));
        });
    }

    static get className() {
        return 'kerasNlp>TransformerEncoderBlock';
    }

    computeOutputShape(inputShape) {
        const outputShape = Array.isArray(inputShape[0]) ? inputShape[0] : [...inputShape];
        if (this._outputRange != null) {
            outputShape[1] = this._outputRange;
        }
        return outputShape;
    }
}

tf.serialization.registerClass(TransformerEncoderBlock);


class EinsumDense extends tf.layers.Layer {

    constructor(config) {
        super(config);
        this.equation = config.equation;
        if (typeof outputShape === 'number') {
            this.partialOutputShape = [config.outputShape];
        } else {
            this.partialOutputShape = [...config.outputShape];
        }
        this.biasAxes = config.biasAxes;
        this.activation = getActivation(config.activation);
        this.kernelInitializer = getInitializer(config.kernelInitializer || 'glorotUniform');
        this.biasInitializer = getInitializer(config.biasInitializer || 'zeros');
        this.kernelRegularizer = getRegularizer(config.kernelRegularizer);
        this.biasRegularizer = getRegularizer(config.biasRegularizer);
        // They forgot about the activity regularizer in Python __init__
        this.activityRegularizer = getRegularizer(config.activityRegularizer);
        this.kernelConstraint = getConstraint(config.kernelConstraint);
        this.biasConstraint = getConstraint(config.biasConstraint);
    }

    build(inputShape) {
        const shapeData = _analyzeEinsumString(this.equation, this.biasAxes, inputShape, this.partialOutputShape);
        const [kernelShape, biasShape, fullOutputShape] = shapeData;
        this.fullOutputShape = fullOutputShape;
        this.kernel = this.addWeight(
            'kernel',
            kernelShape,
            this.dtype,
            this.kernelInitializer,
            this.kernelRegularizer,
            true,
            this.kernelConstraint
        );

        if (biasShape != null) {
            this.bias = this.addWeight(
                'bias',
                biasShape,
                this.dtype,
                this.biasInitializer,
                this.biasRegularizer,
                true,
                this.biasConstraint
            );
        } else {
            this.bias = null;
        }
        super.build(inputShape);
    }

    computeOutputShape() {
        return this.fullOutputShape;
    }

    getConfig() {
        const config = {
            outputShape: this.partialOutputShape,
            equation: this.equation,
            activation: serialize(this.activation),
            biasAxes: this.biasAxes,
            kernelInitializer: serialize(this.kernelInitializer),
            biasInitializer: serialize(this.biasInitializer),
            kernelRegularizer: serialize(this.kernelRegularizer),
            biasRegularizer: serialize(this.biasRegularizer),
            activityRegularizer: serialize(this.activityRegularizer),
            kernelConstraint: serialize(this.kernelConstraint),
            biasConstraint: serialize(this.biasConstraint),
        };
        const baseConfig = super.getConfig();
        return {...baseConfig, config};
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs;
            // TODO: wait until TF.js supports ellipsis notation in einsum eq
            // let ret = tf.einsum(this.equation, inputs, this.kernel.read());
            let ret = einsum(this.equation, inputs, this.kernel.read());
            if (this.bias != null) {
                ret = tf.add(ret, this.bias.read());
            }
            if (this.activation != null) {
                ret = this.activation.apply(ret);
            }
            return ret;
        });
    }

    static get className() {
        return 'EinsumDense';
    }
}

tf.serialization.registerClass(EinsumDense);


function _analyzeEinsumString(equation, biasAxes, inputShape, outputShape) {
    const dotReplacedString = equation.replace(/\.\.\./g, '0');

    let splitString = dotReplacedString.match(/^([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)$/);
    if (splitString) {
        return _analyzeSplitString(splitString, biasAxes, inputShape, outputShape);
    }

    splitString = dotReplacedString.match(/^0([a-zA-Z]+),([a-zA-Z]+)->0([a-zA-Z]+)$/);
    if (splitString) {
        return _analyzeSplitString(splitString, biasAxes, inputShape, outputShape, true);
    }

    splitString = dotReplacedString.match(/^([a-zA-Z]{2,})0,([a-zA-Z]+)->([a-zA-Z]+)0$/);
    if (splitString) {
        return _analyzeSplitString(splitString, biasAxes, inputShape, outputShape);
    }

    throw new Error(`Invalid einsum equation '${equation}'. Equations must be in the form [X],[Y]->[Z], ...[X],[Y]->...[Z], or [X]...,[Y]->[Z]....`);
}


function _analyzeSplitString(splitString, biasAxes, inputShape, outputShape, leftElided = false) {
    const inputSpec = splitString[1];
    const weightSpec = splitString[2];
    const outputSpec = splitString[3];
    const elided = inputShape.length - inputSpec.length;

    if (typeof outputShape === 'number') {
        outputShape = [outputShape];
    } else {
        outputShape = [...outputShape];
    }

    outputShape.unshift(inputShape[0]);

    if (elided > 0 && leftElided) {
        for (let i = 1; i < elided; i++) {
            outputShape.splice(1, 0, inputShape[i]);
        }
    } else if (elided > 0 && !leftElided) {
        for (let i = inputShape.length - elided; i < inputShape.length; i++) {
            outputShape.push(inputShape[i]);
        }
    }

    let inputDimMap, outputDimMap;
    if (leftElided) {
        // Original: negative index, but I don't think that's necessary
        inputDimMap = new Map([...inputSpec].map((dim, i) => [dim, i + elided]));
        outputDimMap = new Map([...outputSpec].map((dim, i) => [dim, i + elided]));
    } else {
        inputDimMap = new Map([...inputSpec].map((dim, i) => [dim, i]));
        outputDimMap = new Map([...outputSpec].map((dim, i) => [dim, i]));
    }

    for (let i = 0; i < inputSpec.length; i++) {
        const dim = inputSpec[i];
        // Original: = inputShape[i], but it should be the mapped inputShape[inputDimMap.get(dim)]
        const inputShapeAtDim = inputShape[inputDimMap.get(dim)];
        if (outputDimMap.has(dim)) {
            const outputShapeAtDim = outputShape[outputDimMap.get(dim)];
            if (outputShapeAtDim != null && outputShapeAtDim !== inputShapeAtDim) {
                throw new Error(`Input shape and output shape do not match at shared dimension '${dim}'. Input shape is ${inputShapeAtDim}, and output shape is ${outputShapeAtDim}.`);
            }
        }
    }

    for (const dim of outputSpec) {
        if (!inputSpec.includes(dim) && !weightSpec.includes(dim)) {
            // Original: % (, , , outputSpec), but the last spec should be weightSpec
            // No wonder this layer is in experimental (facepalm)
            throw new Error(`Dimension '${dim}' was specified in the output '${outputSpec}' but has no corresponding dim in the input spec '${inputSpec}' or weight spec '${weightSpec}.'`);
        }
    }

    const weightShape = [];
    for (const dim of weightSpec) {
        if (inputDimMap.has(dim)) {
            weightShape.push(inputShape[inputDimMap.get(dim)]);
        } else if (outputDimMap.has(dim)) {
            weightShape.push(outputShape[outputDimMap.get(dim)]);
        } else {
            // This message is also wrong: in this case the einsum equation is invalid, never mind specifying anything. Not fixing it
            throw new Error(`Weight dimension '${dim}' did not have a match in either the input spec '${inputSpec}' or the output spec '${outputSpec}'. For this layer, the weight must be fully specified.`);
        }
    }

    let biasShape;
    if (biasAxes != null) {
        const numLeftElided = leftElided ? elided : 0;
        // This map is redundant (idxMap.get(char) === outputShape[outputDimMap.get(char)]) and char is an inconsistent name with dim
        const idxMap = new Map([...outputSpec].map((char, i) => [char, outputShape[i + numLeftElided]]));

        for (const char of biasAxes) {
            if (!outputSpec.includes(char)) {
                throw new Error(`Bias dimension '${char}' was requested, but is not a part of the output specification '${outputSpec}'`);
            }
        }

        const firstBiasLocation = Math.min(...[...biasAxes].map(char => outputSpec.indexOf(char)));
        const biasOutputSpec = outputSpec.slice(firstBiasLocation);

        biasShape = [...biasOutputSpec].map(char => biasAxes.includes(char) ? idxMap.get(char) : 1);

        if (!leftElided) {
            biasShape = [...biasShape, ...new Array(elided).fill(1)];
        }
    } else {
        biasShape = null;
    }

    return [weightShape, biasShape, outputShape];
}


class Gelu extends tf.layers.Layer {

    call(inputs) {
        return tf.tidy(() => {
            const x = Array.isArray(inputs) ? inputs[0] : inputs;
            const coeff = tf.cast(0.044715, x.dtype);
            return x.div(2).mul(
                tf.tanh(
                    tf.pow(x, 3).mul(coeff).add(x).mul(0.7978845608028654)
                ).add(1.0)
            );
        });
    }

    static get className() {
        return 'Text>gelu';
    }
}

tf.serialization.registerClass(Gelu);


export default class QuestionAnswerer {
    constructor(model, tokenizer, isTFLite = false) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.isTFLite = isTFLite;
    }

    static async fromFiles(modelPath, vocabPath, ...args) {
        const modelPromise = this.isTFLite ? tflite.loadTFLiteModel(modelPath) : tf.loadLayersModel(modelPath);
        const tokenizerPromise = Tokenizer.fromVocabFile(vocabPath);
        const model = await modelPromise;
        const tokenizer = await tokenizerPromise;
        return new this(model, tokenizer, ...args);
    }

    async compute(question, context) {
        const {tokens: tokens, tokenTypes: tokenTypes, mask: mask, origs: origs} = this.tokenizer.pack(question, context, true);
        const logits = tf.tidy(() => {
            let results = this.model.predict([
                tf.tensor(tokens, [1, tokens.length], 'int32'),
                tf.tensor(mask, [1, mask.length], 'int32'),
                tf.tensor(tokenTypes, [1, tokenTypes.length], 'int32')
            ]);
            if (this.isTFLite) {
                // FIXME
            }
            return tf.transpose(tf.squeeze(results));
        });
        const logitValues = await logits.array();
        logits.dispose();
        const [startLogits, endLogits] = logitValues.map(startOrEndLogits =>
            // Only preserve the logits that are in the answer
            startOrEndLogits.map((x, i) => [i, x]).slice(tokenTypes.indexOf(1))
        );
        return {startLogits: startLogits, endLogits: endLogits, origs: origs};
    }

    async answer(question, context, threshold, returnScore = false) {
        let bestScore;
        let bestAnswerString;

        if (Array.isArray(context)) {
            answerPromises = context.map(paragraph => this.answer(question, paragraph, null, true));
            for (const promise of answerPromises) {
                const [answerString, score] = await promise;
                if (score >= bestScore || bestScore == null) {
                    bestScore = score;
                    bestAnswerString = answerString;
                }
            }
        } else {
            const {startLogits: startLogits, endLogits: endLogits, origs: origs} = await this.compute(question, context);
            const bestStartLogits = startLogits.sort(([i, x], [j, y]) => x - y).slice(-20);
            const bestEndLogits = endLogits.sort(([i, x], [j, y]) => x - y).slice(-20);

            let bestRange = [];
            for (const [startPos, startLogit] of bestStartLogits) {
                for (const [endPos, endLogit] of bestEndLogits) {
                    if (startPos > endPos) {
                        continue;
                    }

                    const score = startLogit + endLogit;
                    if (score >= bestScore || bestScore == null) {
                        bestScore = score;
                        bestRange = [startPos, endPos + 1];
                    }
                }
            }

            bestAnswerString = origs.slice(...bestRange).join('').trim();
        }

        if (bestScore == null || (typeof threshold === 'number' && bestScore < threshold)) {
            return;
        }

        if (returnScore) {
            return [bestAnswerString, bestScore];
        }
        return bestAnswerString;
    }
}
