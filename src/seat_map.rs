use buckshot_roulette_gameplay_engine::{player_number::PlayerNumber, seat::Seat};

#[derive(Debug, Clone)]
pub struct SeatMap {
    pub own_player: PlayerNumber,
    pub left_player: PlayerNumber,
    pub opposite_player: PlayerNumber,
    pub right_player: PlayerNumber,

    pub own_seat_index: usize,
    pub left_seat_index: Option<usize>,
    pub opposite_seat_index: Option<usize>,
    pub right_seat_index: Option<usize>,
}

fn left_player(player_number: PlayerNumber) -> PlayerNumber {
    match player_number {
        PlayerNumber::One => PlayerNumber::Two,
        PlayerNumber::Two => PlayerNumber::Three,
        PlayerNumber::Three => PlayerNumber::Four,
        PlayerNumber::Four => PlayerNumber::One,
    }
}

fn opposite_player(player_number: PlayerNumber) -> PlayerNumber {
    match player_number {
        PlayerNumber::One => PlayerNumber::Three,
        PlayerNumber::Two => PlayerNumber::Four,
        PlayerNumber::Three => PlayerNumber::One,
        PlayerNumber::Four => PlayerNumber::Two,
    }
}

fn right_player(player_number: PlayerNumber) -> PlayerNumber {
    match player_number {
        PlayerNumber::One => PlayerNumber::Four,
        PlayerNumber::Two => PlayerNumber::One,
        PlayerNumber::Three => PlayerNumber::Two,
        PlayerNumber::Four => PlayerNumber::Three,
    }
}

impl SeatMap {
    pub fn new(own_player: PlayerNumber, seats: &Vec<Seat>) -> Self {
        let left_player = left_player(own_player);
        let right_player = right_player(own_player);
        let opposite_player = opposite_player(own_player);

        let mut own_seat_index = None;
        let mut left_seat_index = None;
        let mut opposite_seat_index = None;
        let mut right_seat_index = None;

        for i in 0..seats.len() {
            let seat_player = seats[i].player_number();
            if seat_player == own_player {
                own_seat_index = Some(i);
            } else if seat_player == left_player {
                left_seat_index = Some(i);
            } else if seat_player == right_player {
                right_seat_index = Some(i);
            } else if seat_player == opposite_player {
                opposite_seat_index = Some(i);
            }
        }
        SeatMap {
            own_player,
            left_player,
            right_player,
            opposite_player,
            own_seat_index: own_seat_index.unwrap(),
            opposite_seat_index: opposite_seat_index,
            left_seat_index,
            right_seat_index,
        }
    }
}
