var express = require('express');
var http = require('http');
var bodyParser = require('body-parser');
var app = express();
var server = http.createServer(app);
var port = 8000;
var fs  = require('fs');
const Promise = require('bluebird');
const cmd = require('node-cmd');
const getAsync = Promise.promisify(cmd.get, { multiArgs: true, context: cmd })

app.use(express.static(__dirname));
app.use(bodyParser.urlencoded({ extended: true }));

app.all('*', function(_, res, next) {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'PUT, GET, POST, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Content-Type');

  next();
});

app.post('/submit', async (req, res) => {

  let { title, question } = req.body;
  let cmd, output = "", error = "", out;
    console.log(question);
    console.log(title);

    const { exec } = require('child_process');

    exec('python model.py '+title, (err, stdout, stderr) => {
      // your callback
      console.log(stdout);
      out=JSON.parse(stdout);
      res.send({
        out
      });
    });


});

app.get('/', function(_, res) {
  res.sendfile(`${__dirname}/views/basic.html`);
});


console.log("Listening at " + port);
server.listen(port);

module.exports = app;
