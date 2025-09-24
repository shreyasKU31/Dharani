function getCurrentCropSeason() {
  const today = new Date();
  const currentMonth = today.getMonth() + 1;

  // Kharif Season (June to October)
  if (currentMonth >= 6 && currentMonth <= 10) {
    return "Kharif";
  }
  // Rabi Season (November to March)
  else if (currentMonth >= 11 || currentMonth <= 3) {
    return "Rabi";
  }
  // The period between Rabi harvest and Kharif sowing (April, May)
  else {
    return "Annual";
  }
}

// --- Example Usage ---
const season = getCurrentCropSeason();
console.log(`The current crop season is: ${season}`);
