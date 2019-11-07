const express = require('express')
var multer = require('multer');
var fs = require('fs'); 
const spawn = require("child_process").spawn;

const app = express()
const port = 3000

const FILES_DIR = '/param_files/'
var upload = multer({ dest: FILES_DIR});

var one_hot_state = 0b000;
var N_partitions = 3;

app.get('/', (req, res) => res.send('Federated Learning - Server'))

// File input field name is simply 'file'
app.post('/upload_local_param', upload.single('file'), function(req, res) {
    var node_n = req.query.node_n;
    one_hot_state = one_hot_state | (1<<node_n)
    var file_destination = __dirname + '/param_files/' + req.file.originalname;
    const pythonProcess = spawn('python',['federated_server.py']);
    fs.rename(req.file.path, file_destination, function(err) {
        if (err) {
            console.log(err);
            res.sendStatus(500);
        } else {
            res.json({
                message: 'File uploaded successfully',
                filename: req.file.originalname
            });
        }
    });
});

app.get('/fed_avg_done', function(req, res) {
    status = req.query.status
    console.log(status)
})

var base_path = __dirname + FILES_DIR;
app.get('/get_global_param', function(req, res) {
    filename = req.query.filename
    res.download(base_path+filename, filename, function (err) {
        if (err) {
            console.log(err);
            res.sendStatus(500);
        } else {
            console.log("Successful Download")
        }
    })
})

app.listen(port, () => console.log(`Example app listening on port ${port}!`))
