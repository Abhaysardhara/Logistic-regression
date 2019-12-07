const x_vals = [];
const y_vals = [];

let a, b, c, d;
let dragging = false;

const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

function setup() {
    createCanvas(800, 800);
    a = tf.variable(tf.scalar(random(-1, 1)));
    b = tf.variable(tf.scalar(random(-1, 1)));
    c = tf.variable(tf.scalar(random(-1, 1)));
    d = tf.variable(tf.scalar(random(-1, 1)));


}

function mousePressed() {
    dragging = true;
}

function mouseReleased() {
    dragging = false;
}

function loss(pred, labels) {
    return pred.sub(labels).square().mean();
}

function predict(x) {
    let xs = tf.tensor1d(x);
    // ys = xs.mul(m).add(b);
    // ax^3 + bX^2 + cx + d
    ys = xs.pow(tf.scalar(3)).mul(a).add(xs.square().mul(b)).add(xs.mul(c)).add(d);
    return ys;
}


function draw() {

    if(dragging) {
        let x = map(mouseX, 0, width, -1, 1);
        let y = map(mouseY, 0, height, 1, -1);
        x_vals.push(x);
        y_vals.push(y);
    } 
    else {
        tf.tidy(() => {
            if (x_vals.length > 0) {
                const ys = tf.tensor1d(y_vals);
                optimizer.minimize(() => loss(predict(x_vals), ys));
            }
        });
    }

        background(0);
        stroke(255);
        strokeWeight(8);
        for(let i =0; i < x_vals.length; i++) 
        {
            let px = map(x_vals[i], -1, 1, 0, width);
            let py= map(y_vals[i], -1, 1, height, 0);
            point(px, py);
        }

        const curveX = [];
        for(let i= -1; i < 1.01; i += 0.05) {
             curveX.push(i);
        }

        tf.tidy(() => {
            // const lineX = [0, 1];
            const ys = tf.tidy(() => predict(curveX));
            let curveY = ys.dataSync();
            ys.dispose();

            // let x1 = map(lineX[0], -1, 1, 0, width);
            // let x2 = map(lineX[1], -1, 1, 0, width);
            beginShape();
            noFill();
            stroke(255);
            strokeWeight(2);

            for(let i = 0; i < curveX.length; i++) {
                let x = map(curveX[i], -1, 1, 0, width);
                let y = map(curveY[i], -1, 1, height, 0);
                vertex(x, y);
            }
            endShape();

            // let lineY = ys.dataSync();
            // let y1 = map(lineY[0], -1, 1, height, 0);
            // let y2 = map(lineY[1], -1, 1, height, 0);

            // line(x1, y1, x2, y2);
        });
    console.log(tf.memory().numTensors);
    // ys.dispose();
}