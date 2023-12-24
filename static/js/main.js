let offset = 0; // смещение от левого края
const sliderLine = document.querySelector('.slider_line');
const button1 = document.querySelector('.slider_next_1');
const button2 = document.querySelector('.slider_next_2');
const button3 = document.querySelector('.slider_next_3');
const button4 = document.querySelector('.slider_next_4');
const main = document.querySelector('.main');
const footer = document.querySelector('.footer');
document.querySelector('.slider_next_1').addEventListener('click', function() {
    sliderLine.style.left = 0 + 'px';
    button1.style.backgroundColor = 'rgb(0, 47, 255)';
    button2.style.backgroundColor = 'rgb(255, 255, 255)';
    button3.style.backgroundColor = 'rgb(255, 255, 255)';
    button4.style.backgroundColor = 'rgb(255, 255, 255)';
});

document.querySelector('.slider_next_2').addEventListener('click', function() {
    sliderLine.style.left = -898 + 'px';
    button1.style.backgroundColor = 'rgb(255, 255, 255)';
    button2.style.backgroundColor = 'rgb(0, 47, 255)';
    button3.style.backgroundColor = 'rgb(255, 255, 255)';
    button4.style.backgroundColor = 'rgb(255, 255, 255)';
});

document.querySelector('.slider_next_3').addEventListener('click', function() {
    offset += 900;
    sliderLine.style.left = -1800 + 'px';
    button1.style.backgroundColor = 'rgb(255, 255, 255)';
    button2.style.backgroundColor = 'rgb(255, 255, 255)';
    button3.style.backgroundColor = 'rgb(0, 47, 255)';
    button4.style.backgroundColor = 'rgb(255, 255, 255)';
});

document.querySelector('.slider_next_4').addEventListener('click', function() {
    sliderLine.style.left = -2724 + 'px';
    button1.style.backgroundColor = 'rgb(255, 255, 255)';
    button2.style.backgroundColor = 'rgb(255, 255, 255)';
    button3.style.backgroundColor = 'rgb(255, 255, 255)';
    button4.style.backgroundColor = 'rgb(0, 47, 255)';
});

function init() {
    const height = document.documentElement.offsetHeight;
    console.log(height);
    if (height > 1450) {
        footer.style.top = height - 170 + 'px';
        window.addEventListener('resize', init);
    } else {
        footer.style.top = 1480 + 'px';
        window.addEventListener('resize', init);
    }
    
}
init();
