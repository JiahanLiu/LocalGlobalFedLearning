const express = require('express')
var multer = require('multer');
var fs = require('fs'); 
const spawn = require("child_process").spawn;

const app = express()
const port = 3000

const FILES_DIR = '/param_files/'
var upload = multer({ dest: FILES_DIR});

const LOCAL_ALL_DONE = 0b111;
const LOCAL_RESET = 0b000;

var one_hot_state = 0b000;
var fed_avg_done_flag = 0b0;
var global_sync_done_flag = 0b1; 

app.get('/', (req, res) => res.send('Federated Learning - Server'))

// File input field name is simply 'file'
app.post('/upload_local_param', upload.single('file'), function(req, res) {
    var file_destination = __dirname + '/param_files/' + req.file.originalname;
    var node_n = req.query.node_n;
    one_hot_state = one_hot_state | (1<<node_n)
    // one_hot_state = LOCAL_ALL_DONE
    if(LOCAL_ALL_DONE == one_hot_state){
        one_hot_state = LOCAL_RESET;
        global_sync_done_flag = 0b0;
        const pythonProcess = spawn('python3',['../federated_server.py']);
    }
    fs.rename(req.file.path, file_destination, function(err) {
        if (err) {
            console.log(err);
            res.sendStatus(500);
        } else {
            console.log("Yes, Upload from: " + node_n);
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
    if(status == "success") {
        fed_avg_done_flag = 0b1;
        res.sendStatus(200)
    } else {
        res.sendStatus(500)
    }
})

var base_path = __dirname + FILES_DIR;
app.get('/get_global_param', function(req, res) {
    var filename = req.query.filename;
    var node_n = req.query.node_n;
    // one_hot_state = LOCAL_ALL_DONE;
    one_hot_state = one_hot_state | (1<<node_n)
    if(LOCAL_ALL_DONE == one_hot_state){
        one_hot_state = LOCAL_RESET
        fed_avg_done_flag = 0b0;
        global_sync_done_flag = 0b1;
    }
    res.download(base_path+filename, filename, function (err) {
        if (err) {
            console.log(err);
            res.sendStatus(500);
        } else {
            console.log("Yes, Download")
        }
    })
})

app.get('/query_global_sync_done', function(req, res) {
    res.json({
        status: global_sync_done_flag
    });
})

app.get('/query_fed_avg_done', function(req, res) {
    res.json({
        status: fed_avg_done_flag
    });
})

app.listen(port, () => console.log(`Example app listening on port ${port}!`))
