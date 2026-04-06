# DART-GUI systemd services

Two systemd units that run the local OSWorld desktop environment server,
which the validation runner (`run.py`) connects to on port 4999.

## Units

### `xvfb.service`
Starts an Xvfb virtual framebuffer on display `:99` (1920x1080x24).
Required by the desktop server for screenshots and pyautogui actions.

### `dart-desktop-server.service`
Runs `validation/local_desktop_server.py` on port 4999.
Depends on `xvfb.service` — systemd starts Xvfb first automatically.

## Install & enable

```bash
sudo cp /mnt/dart-gui/services/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now xvfb dart-desktop-server
```

## Check status

```bash
systemctl status xvfb dart-desktop-server
curl -s http://localhost:4999/server/list
```

## Disable

```bash
sudo systemctl disable --now xvfb dart-desktop-server
```
