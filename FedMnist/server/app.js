const express = require('express')
var multer = require('multer');
var fs = require('fs'); 

const app = express()
const port = 3000

const FILES_DIR = '/param_files/'
var upload = multer({ dest: FILES_DIR});

app.get('/', (req, res) => res.send('Federated Learning - Server'))

// File input field name is simply 'file'
app.post('/upload_local_param', upload.single('file'), function(req, res) {
    var file_destination = __dirname + '/param_files/' + req.file.originalname;
    fs.rename(req.file.path, file_destination, function(err) {
        if (err) {
            console.log(err);
            res.send(500);
        } else {
            res.json({
                message: 'File uploaded successfully',
                filename: req.file.originalname
            });
        }
    });
});

var GLOBAL_PARAMS_FILE = 'node0_local_param.txt'
var GLOBAL_PARAMS_PATH = __dirname + FILES_DIR + GLOBAL_PARAMS_FILE;
app.get('/get_global_param', function(req, res) {
    res.download(GLOBAL_PARAMS_PATH, GLOBAL_PARAMS_FILE, function (err) {
        if (err) {
            console.log(err);
            res.send(500);
        } else {
            console.log("Successful Download")
        }
    })
})

app.listen(port, () => console.log(`Example app listening on port ${port}!`))
