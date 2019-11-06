const express = require('express')
var multer = require('multer');
var fs = require('fs'); 

const app = express()
const port = 3000

const FILES_DIR = '/local_uploads/'
var upload = multer({ dest: FILES_DIR});

app.get('/', (req, res) => res.send('Hello World!'))

// File input field name is simply 'file'
app.post('/local_upload', upload.single('file'), function(req, res) {
    var file = __dirname + '/local_uploads/' + req.file.originalname;
    fs.rename(req.file.path, file, function(err) {
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

var GLOBAL_PARAMS_FILE = 'file.txt'
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
