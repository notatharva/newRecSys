var util = require("util");
var song = "Break Through - Colbie Caillat";
var spawn = require("child_process").spawn;
var process = spawn('python',["songs.py",song]);

util.log('readingin')

process.stdout.on('data',function(chunk){

    var textChunk = chunk.toString('utf8');// buffer to string

    util.log(textChunk);
});