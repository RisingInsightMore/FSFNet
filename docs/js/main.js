// FSFNet project page - tab switching and navbar highlighting

// Tab switching. Each tab set lives inside its own .tab-group, so several
// independent tab sets can coexist in one section (e.g. the SOTA comparison
// tabs and the ablation tabs, both inside #results).
document.querySelectorAll('.tab-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
        var group = this.closest('.tab-group');
        if (!group) return;
        group.querySelectorAll('.tab-btn').forEach(function (b) {
            b.classList.remove('active');
            b.setAttribute('aria-selected', 'false');
        });
        group.querySelectorAll('.tab-content').forEach(function (c) { c.classList.remove('active'); });
        this.classList.add('active');
        this.setAttribute('aria-selected', 'true');
        var target = group.querySelector('.tab-content[id="tab-' + this.dataset.tab + '"]');
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
