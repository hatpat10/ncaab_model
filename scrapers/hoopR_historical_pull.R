# scrapers/hoopR_historical_pull.R
library(hoopR)
library(toRvik)
library(cbbdata)
library(dplyr)
library(arrow)
library(purrr)

# Create output directory if it doesn't exist
dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)

# 1. Team box scores (seasons 2020-2025)
message("Pulling team box scores...")
progressr::with_progress({
  team_box <- hoopR::load_mbb_team_box(seasons = 2020:2025)
})
cat(nrow(team_box), "rows from", n_distinct(team_box$game_id), "games\n")

# 2. Player box scores
message("Pulling player box scores...")
progressr::with_progress({
  player_box <- hoopR::load_mbb_player_box(seasons = 2020:2025)
})
cat(nrow(player_box), "rows\n")

# 3. Schedule
message("Pulling schedule...")
progressr::with_progress({
  schedule <- hoopR::load_mbb_schedule(seasons = 2020:2025)
})
cat(nrow(schedule), "rows\n")

# 4. BartTorvik ratings
# toRvik works for 2020-2023, cbbdata for 2024-2025
message("Pulling BartTorvik ratings 2020-2023 via toRvik...")
bt_2020_2023 <- purrr::map_dfr(2020:2023, ~toRvik::bart_ratings(year = .x))
cat(nrow(bt_2020_2023), "rows for 2020-2023\n")

message("Pulling BartTorvik ratings 2024-2025 via cbbdata...")
bt_2024 <- cbbdata::cbd_torvik_ratings(year = 2024)
bt_2025 <- cbbdata::cbd_torvik_ratings(year = 2025)
bt_2024_2025 <- dplyr::bind_rows(bt_2024, bt_2025)
cat(nrow(bt_2024_2025), "rows for 2024-2025\n")

# 5. Save all to parquet
message("Saving to parquet...")
arrow::write_parquet(team_box,      "data/raw/hoopR_team_box_2020_2025.parquet")
arrow::write_parquet(player_box,    "data/raw/hoopR_player_box_2020_2025.parquet")
arrow::write_parquet(schedule,      "data/raw/hoopR_schedule_2020_2025.parquet")
arrow::write_parquet(bt_2020_2023,  "data/raw/barttorvik_2020_2023.parquet")
arrow::write_parquet(bt_2024_2025,  "data/raw/barttorvik_2024_2025.parquet")

message("Done! All historical data saved to data/raw/")