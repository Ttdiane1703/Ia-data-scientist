document.addEventListener('DOMContentLoaded', () => {
  const authScreen = document.getElementById('auth-screen');
  const appShell = document.getElementById('app-shell');

  if (appShell) {
    appShell.hidden = false;
  }

  if (authScreen) {
    authScreen.hidden = true;
  }

  const savedTheme = localStorage.getItem('ia_ds_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);

  const firstPage = document.querySelector('.nav-item[data-page="dataset"]');
  if (firstPage) {
    firstPage.classList.add('is-active');
    const targetPage = document.querySelector('.page[data-page="dataset"]');
    if (targetPage) targetPage.classList.add('active');
  }
});
