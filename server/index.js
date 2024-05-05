const express = require("express");
const app = express();
const cors = require("cors");
const fs = require("fs");
const path = require("path");

const databasePath = path.join(__dirname, "database.json");

function loadDatabase() {
  try {
    const data = fs.readFileSync(databasePath);
    return JSON.parse(data);
  } catch (error) {
    console.error("Error loading database:", error);
    return { users: [] }; // Return an empty database if file doesn't exist or has errors
  }
}

function saveDatabase(data) {
  try {
    fs.writeFileSync(databasePath, JSON.stringify(data, null, 2));
  } catch (error) {
    console.error("Error saving database:", error);
  }
}

let database = loadDatabase();

app.use(cors());
app.use(express.json());

app.get("/slots", (req, res) => {
  res.json({ slots : database.available_slots - database.occupied_slots });
});

app.post("/slots/:username", (req, res) => {
  const username = req.params.username;
  const body = req.body;

  if (database.available_slots > 0) {
    if (!database.users[username]) {
      database.users[username] = [];
    }
    database.users[username].push(body);
    database.occupied_slots = database.occupied_slots+1;
    saveDatabase(database);
    res.json({ slots : database.available_slots - database.occupied_slots , message: "slot booked successfully" });
  } else res.json({ slots :database.available_slots - database.occupied_slots , message: "Not able to book any more tickets" });
});

app.get("/bookings/:username", (req, res) => {
  const username = req.params.username;

  const bookingsbyUsername = database.users[username];

  res.json({ bookings : bookingsbyUsername });
});

app.post("/model/slotcount",(req,res)=>{
  let new_slots = req.body.empty_slots;
  if(database.available_slots > new_slots) database.occupied_slots = database.occupied_slots - (database.available_slots - new_slots);
  database.available_slots = req.body.empty_slots
  saveDatabase(database);

  res.json({"message":"update successfull"});
});

app.listen(8000, () => {
  console.log("server started successfylly");
});
