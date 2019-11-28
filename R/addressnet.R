encoding <- list(
  '0' = 1,
  '1' = 2,
  '2' = 3,
  '3' = 4,
  '4' = 5,
  '5' = 6,
  '6' = 7,
  '7' = 8,
  '8' = 9,
  '9' = 10,
  'a' = 11,
  'b' = 12,
  'c' = 13,
  'd' = 14,
  'e' = 15,
  'f' = 16,
  'g' = 17,
  'h' = 18,
  'i' = 19,
  'j' = 20,
  'k' = 21,
  'l' = 22,
  'm' = 23,
  'n' = 24,
  'o' = 25,
  'p' = 26,
  'q' = 27,
  'r' = 28,
  's' = 29,
  't' = 30,
  'u' = 31,
  'v' = 32,
  'w' = 33,
  'x' = 34,
  'y' = 35,
  'z' = 36,
  '!' = 37,
  '"' = 38,
  '#' = 39,
  '$' = 40,
  '%' = 41,
  '&' = 42,
  "'" = 43,
  '(' = 44,
  ')' = 45,
  '*' = 46,
  '+' = 47,
  ',' = 48,
  '-' = 49,
  '.' = 50,
  '/' = 51,
  ':' = 52,
  ';' = 53,
  '<' = 54,
  '=' = 55,
  '>' = 56,
  '?' = 57,
  '@' = 58,
  '[' = 59,
  '\\' = 60,
  ']' = 61,
  '^' = 62,
  '_' = 63,
  '`' = 64,
  '{' = 65,
  '|' = 66,
  '}' = 67,
  '~' = 68,
  ' ' = 69
)

#' Encodes text for input into the neural network
#' @param txt a vector of address strings
#' @return a list containing the original string lengths and a padded matrix encoding the addresses
encode_text <- function(txt) {
  n_lines <- length(txt)
  lengths <- nchar(txt)
  longest_string <- max(lengths)
  if (n_lines == 1) {
    lengths <- list(lengths)
  }

  txt <-
    stringr::str_pad(txt, longest_string, side = "right", pad = "0")
  list(
    lengths = tensorflow::tf$constant(lengths, dtype = tensorflow::tf$int64),
    text = tensorflow::tf$constant(
      matrix(
        unlist(sapply(strsplit(txt, split = ""), function(char) {
          encoding[char]
        })),
        nrow = n_lines,
        ncol = longest_string,
        byrow = TRUE
      ),
      dtype = tensorflow::tf$int64
    )
  )
}

#' Loads the tensorflow model
#' @return the tensorflow serving model
load_model <- function() {
  m <- reticulate::py_suppress_warnings({
    tensorflow::tf$compat$v2$saved_model$load(file.path("data", "1574752734/"))
  })
  m$signatures["serving_default"]
}

#' Converts a vector of addresses into a table showing each address component
#' @param address a vector of addresses
#' @return a tibble of address components
#' @import dplyr
#' @import tidyr
#' @export
parse_addresses <- function(address) {
  model <- load_model()
  data <- encode_text(tolower(address))
  result <-
    model(encoded_text = data$text, lengths = data$lengths)$class_ids$numpy()

  result <- lapply(seq(length(address)), function(i) {
    processed_address <-
      lapply(seq(nchar(address[i])), function(j) {
        list(
          char = substr(address[i], j, j),
          cls = result[i, j],
          address_index = i,
          char_index = j
        )
      })
    return(
      dplyr::bind_rows(processed_address) %>%
        dplyr::mutate(prev_cls = dplyr::lag(cls)) %>%
        dplyr::mutate(new_grp = tidyr::replace_na(cls != prev_cls, TRUE)) %>%
        dplyr::mutate(grp_ticker = cumsum(new_grp))
    )
  })
  dplyr::bind_rows(result) %>%
    dplyr::group_by(cls, address_index, grp_ticker) %>%
    dplyr::mutate(address_parts = toupper(paste0(char, collapse = ""))) %>%
    dplyr::ungroup() %>%
    dplyr::select(address_index, cls, address_parts) %>%
    dplyr::distinct(address_index, cls, .keep_all = TRUE) %>%
    dplyr::filter(cls > 0) %>%
    dplyr::mutate(
      cls = dplyr::recode(
        cls,
        `1` = "building_name",
        `2` = "level_number_prefix",
        `3` = "level_number",
        `4` = "level_number_suffix",
        `5` = "level_type",
        `6` = "flat_number_prefix",
        `7` = "flat_number",
        `8` = "flat_number_suffix",
        `9` = "flat_type",
        `10` = "number_first_prefix",
        `11` = "number_first",
        `12` = "number_first_suffix",
        `13` = "number_last_prefix",
        `14` = "number_last",
        `15` = "number_last_suffix",
        `16` = "street_name",
        `17` = "street_suffix",
        `18` = "street_type",
        `19` = "locality_name",
        `20` = "state",
        `21` = "postcode"
      )
    ) %>%
    tidyr::spread(cls, address_parts)
}
