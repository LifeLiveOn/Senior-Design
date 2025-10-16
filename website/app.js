const express = require("express");
const path = require("path");

const app = express();
const port = 3000;

// Prints out all urls
app.use((req, res, next) => {
   console.log(req.url, req.method);
   next();
});

// Parses all json request bodies
// app.use(express.json());

// Access files from public folder
app.use(express.static(path.join(__dirname, 'public')));

// Sets server port
app.listen(port);