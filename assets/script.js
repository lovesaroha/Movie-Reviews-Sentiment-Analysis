"use-strict";
/*  Love Saroha
    lovesaroha1994@gmail.com (email address)
    https://www.lovesaroha.com (website)
    https://github.com/lovesaroha  (github)
*/

// Themes.
const themes = [{ normal: "#5468e7", light: "#6577e9", veryLight: "#eef0fd" }, { normal: "#e94c2b", light: "#eb5e40", veryLight: "#fdedea" }];

// Choose random color theme.
let colorTheme = themes[Math.floor(Math.random() * themes.length)];

// This function set random color theme.
function setTheme() {
  // Change css values.
  document.documentElement.style.setProperty("--primary", colorTheme.normal);
}

// Set random theme.
setTheme();

// Meta data.
let indexFrom = metaData['index_from'];
let maxLen = metaData['max_len'];
let wordIndex = metaData['word_index'];
let vocabularySize = metaData['vocabulary_size'];

let model;
// Load trained model.
tf.loadLayersModel("https://models.lovesaroha.com/Movie-Reviews-Sentiment-Analysis-Model/model.json").then(savedModel => {
  model = savedModel;
  document.getElementById("view_id").innerHTML = document.getElementById("homePage_id").innerHTML;
}).catch(e => { console.log(e); })

// This function take input from user and show score.
function getSentiment(el) {
  if (el.value.length == 0) {
    document.getElementById("progress_id").style = `width: 0%;`;
    document.getElementById("score_id").innerHTML = `0% Positive`;
    return;
  }
  let score = predict(el.value);
  document.getElementById("progress_id").style = `width: ${score * 100}%;`;
  document.getElementById("score_id").innerHTML = `${(score * 100).toFixed(2)}% Positive`;
}

// This function predict score.
function predict(text) {
  const inputText = text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');
  const sequence = inputText.map(word => {
    let wi = wordIndex[word] + indexFrom;
    if (wi > vocabularySize) {
      wordIndex = OOV_INDEX;
    }
    return wi;
  });
  const paddedSequence = padSequences([sequence], maxLen);
  const input = tf.tensor2d(paddedSequence, [1, maxLen]);
  const predictOut = model.predict(input);
  return predictOut.dataSync()[0];
}

const PAD_INDEX = 0;
const OOV_INDEX = 2;
// Padding.
function padSequences(sequences, maxLen, padding = 'pre', truncating = 'pre', value = PAD_INDEX) {
  return sequences.map(seq => {
    // Perform truncation.
    if (seq.length > maxLen) {
      if (truncating === 'pre') {
        seq.splice(0, seq.length - maxLen);
      } else {
        seq.splice(maxLen, seq.length - maxLen);
      }
    }
    // Perform padding.
    if (seq.length < maxLen) {
      const pad = [];
      for (let i = 0; i < maxLen - seq.length; ++i) {
        pad.push(value);
      }
      if (padding === 'pre') {
        seq = pad.concat(seq);
      } else {
        seq = seq.concat(pad);
      }
    }

    return seq;
  });
}