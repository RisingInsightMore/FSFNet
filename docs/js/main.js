// FSFNet project page - tab switching and navbar highlighting

// Tab switching (dataset comparison tables)
document.querySelectorAll('.tab-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
        var group = this.closest('.section');
        group.querySelectorAll('.tab-btn').forEach(function (b) { b.classList.remove('active'); });
        group.querySelectorAll('.tab-content').forEach(function (c) { c.classList.remove('active'); });
        this.classList.add('active');
        var target = document.getElementById('tab-' + this.dataset.tab);
        if (target) target.classList.add('active');
    });
});

// Navbar active link on scroll
var sections = document.querySelectorAll('.section[id]');
var navLinks = document.querySelectorAll('.navbar .nav-links a');

function setActiveNav() {
    var pos = window.scrollY + 90;
    sections.forEach(function (sec) {
        if (sec.offsetTop <= pos && sec.offsetTop + sec.offsetHeight > pos) {
            navLinks.forEach(function (a) { a.classList.remove('active'); });
            var link = document.querySelector('.navbar .nav-links a[href="#' + sec.id + '"]');
            if (link) link.classList.add('active');
        }
    });
}

window.addEventListener('scroll', setActiveNav);
setActiveNav();
