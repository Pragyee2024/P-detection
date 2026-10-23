function showScreen(name) {
  document.querySelectorAll('.screen').forEach(s => s.hidden = true);
  document.getElementById('screen-' + name).hidden = false;
}

// Temporary wiring so you can click through every screen
document.getElementById('btn-submit').onclick = () => showScreen('processing');
document.getElementById('btn-cancel').onclick = () => showScreen('record');
document.getElementById('btn-again').onclick = () => showScreen('record');
document.getElementById('btn-retry').onclick = () => showScreen('record');

// Temporary: click the title on Processing to see Results, the spinner to see Error
document.querySelector('#screen-processing p').onclick = () => showScreen('results');
document.querySelector('.spinner').onclick = () => showScreen('error');

showScreen('record');