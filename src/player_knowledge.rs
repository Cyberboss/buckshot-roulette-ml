use std::collections::VecDeque;

use buckshot_roulette_gameplay_engine::{
    loadout::MAX_SHELLS, shell::ShellType, turn::LearnedShell,
};

use crate::seat_map::SeatMap;

const SHELL_GONE: f64 = 0.0;
const SHELL_UNKNOWN: f64 = 1.0;
const SHELL_BLANK: f64 = 2.0;
const SHELL_LIVE: f64 = 3.0;

#[derive(Debug, Clone)]
pub enum ShellUpdate {
    Learned(LearnedShell),
    Global(GlobalShellUpdate),
}

#[derive(Debug, Clone, Copy)]
pub enum GlobalShellUpdate {
    Ejected(ShellType),
    Inverted,
}

#[derive(Debug, Clone)]
pub struct PlayerKnowledge {
    pub shells: VecDeque<f64>,
    pub remaining_live_rounds: i32,
    pub seat_map: SeatMap,
}

impl PlayerKnowledge {
    pub fn new(seat_map: SeatMap) -> Self {
        PlayerKnowledge {
            shells: VecDeque::with_capacity(MAX_SHELLS.try_into().unwrap()),
            remaining_live_rounds: 0,
            seat_map,
        }
    }

    pub fn initialize(&mut self, total_shells: usize, live_shells: usize) {
        self.shells.clear();
        for _ in 0..total_shells {
            self.shells.push_back(SHELL_UNKNOWN);
        }

        let max_shells: usize = MAX_SHELLS.try_into().unwrap();
        for _ in 0..(max_shells - total_shells) {
            self.shells.push_back(SHELL_GONE);
        }

        self.remaining_live_rounds = live_shells.try_into().unwrap();
    }

    pub fn update(&mut self, update: ShellUpdate) {
        match update {
            ShellUpdate::Global(update) => match update {
                GlobalShellUpdate::Ejected(shell_type) => {
                    self.shells.pop_front();
                    self.shells.push_back(SHELL_GONE);
                    if shell_type == ShellType::Live {
                        self.remaining_live_rounds -= 1;
                    }
                }
                GlobalShellUpdate::Inverted => match self.shells[0] {
                    SHELL_BLANK => self.shells[0] = SHELL_LIVE,
                    SHELL_LIVE => self.shells[0] = SHELL_BLANK,
                    SHELL_UNKNOWN => {}
                    _ => unreachable!(),
                },
            },
            ShellUpdate::Learned(learned_shell) => {
                self.shells[learned_shell.relative_index] = match learned_shell.shell_type {
                    ShellType::Live => SHELL_LIVE,
                    ShellType::Blank => SHELL_BLANK,
                }
            }
        }
    }
}
